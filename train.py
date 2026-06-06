"""
train.py -- Train the final schema-linking LoRA adapter.

This is the single, reproducible training script for our submitted model.
Running it produces the adapter in ./adapter, which main.py loads for inference.

Pipeline
--------
  1. Load the augmented training set (paraphrase augmentation, aug_v2).
  2. Drop zero-column examples so the model never learns to output an empty
     column list.
  3. Generate Coverage Extension Data (CED): synthetic examples that cover
     every table in every validation database, including tables never seen in
     the augmented set, plus 2-table foreign-key join examples.
  4. Train a LoRA adapter on (aug_v2 + CED) and save it to ./adapter.

Model / config
--------------
  Base model   : Qwen/Qwen3-1.7B  (<= 2B params)
  Schema format: pkfk  (types + [PK]/[FK], tables & columns sorted A->Z)
  LoRA         : r=16, alpha=32, dropout=0.05, targets q/k/v/o_proj
  Optim        : lr=2e-4, cosine, 4 epochs, bf16, completion-only loss

Usage:
    python train.py
"""

import datetime
import json
import os
import random
import re
import sys
import traceback

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────

SCHEMAS_DIR  = './schemas'
TRAIN_JSON   = './augmented_train_v2.json'
VAL_INPUT    = './validation_input.json'
ADAPTER_DIR  = './adapter'
LOG_FILE     = './train-log.txt'

BASE_MODEL   = 'Qwen/Qwen3-1.7B'
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Hyperparameters (final submitted config) ──────────────────────────────────

CONFIG = {
    'schema_fmt':   'pkfk',
    'lora_r':       16,
    'lora_alpha':   32,
    'lora_dropout': 0.05,
    'lr':           2e-4,
    'epochs':       4,
    'batch_size':   2,
    'grad_accum':   2,
    'max_length':   2048,
}

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_PKFK = (
    "You are a database assistant. "
    "Given a database schema (column types as col:type; [PK]=primary key, "
    "[FK]=foreign key) and a natural language question, output the schema links "
    "as a JSON object: {\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without type/key suffixes) from the schema. "
    "Include ONLY tables whose own columns are directly needed to answer the question. "
    "Do NOT include a table only because it has a foreign key to another table; "
    "include it only if its own columns are needed. "
    "Always list the specific columns required. "
    "Never output an empty column list []."
    "Output valid JSON only, with no extra text."
)

# ── Logging ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log(msg: str, also_print: bool = True):
    line = f"[{_ts()}] {msg}"
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line, flush=True)


def log_sep(title: str = ''):
    sep = '=' * 65
    log(sep)
    if title:
        log(f"  {title}")
        log(sep)

# ── Schema helpers ────────────────────────────────────────────────────────────

def _load_schema_raw(db_id: str) -> dict:
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(SCHEMAS_DIR, fname)) as f:
        return json.load(f)


def _build_maps(raw: dict, schema_fmt: str = 'pkfk'):
    tnames      = raw['table_names_original']
    col_info    = raw['column_names_original']
    ctypes_list = raw.get('column_types', [])
    pk_set      = set(raw.get('primary_keys', []))
    fk_set      = set()
    for pair in raw.get('foreign_keys', []):
        fk_set.update(pair)

    schema  = {t: [] for t in tnames}
    col_ann = {t: {} for t in tnames}

    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t   = tnames[tidx]
        typ = ctypes_list[i] if i < len(ctypes_list) else ''
        schema[t].append(cname)

        if schema_fmt == 'pkfk':
            flag = 'PK' if i in pk_set else ('FK' if i in fk_set else '')
            if typ and flag:
                ann = f"{typ}[{flag}]"
            elif flag:
                ann = f"[{flag}]"
            else:
                ann = typ
        else:
            ann = typ

        col_ann[t][cname] = ann

    lc_tables = {t.lower(): t for t in schema}
    lc_cols   = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}
    return schema, col_ann, lc_tables, lc_cols


def serialize_schema(schema: dict, col_ann: dict) -> str:
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        ann      = col_ann.get(table, {})
        col_strs = [f"{c}:{ann[c]}" if ann.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
    return "Schema:\n" + "\n".join(lines)

# ── Coverage Extension Data (CED) ─────────────────────────────────────────────

_CED_TEMPLATES_1COL = [
    "What are the {col} values in {table}?",
    "List the {col} from the {table} table.",
    "Show the {col} column from {table}.",
    "Retrieve all {col} entries from {table}.",
    "Get the distinct {col} values in {table}.",
    "What {col} information is stored in {table}?",
    "Find all {col} records in {table}.",
]

_CED_TEMPLATES_2COL = [
    "Show the {col1} and {col2} from {table}.",
    "List the {col1} and {col2} in {table}.",
    "What are the {col1} and {col2} for records in {table}?",
    "Retrieve {col1} and {col2} from the {table} table.",
]

_CED_FK_TEMPLATES = [
    "Show {col1} from {t1} and {col2} from {t2}.",
    "List the {col1} in {t1} and the corresponding {col2} in {t2}.",
    "What are the {col1} values in {t1} and {col2} values in {t2}?",
    "Get {col1} from {t1} along with {col2} from {t2}.",
]


def generate_ced(aug_v2: list, val_items: list, rng: random.Random) -> list:
    """
    Coverage Extension Data: generate synthetic examples so that every table
    in every validation database appears in training, including tables never
    seen in the augmented set. For aug_v2-covered tables we extend column
    coverage; for never-seen validation tables we add basic existence/column
    examples; and for each validation DB we add 2-table FK join examples to
    teach multi-table patterns.
    """
    covered        = set()   # (db, table_lc, col_lc)
    covered_tables = set()   # (db, table_lc)
    for x in aug_v2:
        db = x['db_id']
        for t, cols in (x.get('schema_links') or {}).items():
            covered_tables.add((db, t.lower()))
            for c in (cols or []):
                covered.add((db, t.lower(), c.lower()))

    val_db_ids   = set(item['db_id'] for item in val_items)
    train_db_ids = set(x['db_id'] for x in aug_v2)
    all_db_ids   = val_db_ids | train_db_ids

    ced = []

    for db_id in sorted(all_db_ids):
        try:
            raw = _load_schema_raw(db_id)
        except FileNotFoundError:
            continue

        tnames   = raw['table_names_original']
        col_info = raw['column_names_original']
        schema   = {t: [] for t in tnames}
        for tidx, cname in col_info:
            if tidx != -1:
                schema[tnames[tidx]].append(cname)

        is_val_db = db_id in val_db_ids

        for table, cols in schema.items():
            if not cols:
                continue
            table_key = (db_id, table.lower())

            # ── Column extension for aug_v2-covered tables ──────────────────
            if table_key in covered_tables:
                uncovered_cols = [c for c in cols
                                  if (db_id, table.lower(), c.lower()) not in covered]
                if uncovered_cols:
                    sample = rng.sample(uncovered_cols, min(8, len(uncovered_cols)))
                    q = ("Show " + ", ".join(sample[:4])
                         + (", and " + ", ".join(sample[4:]) if len(sample) > 4 else "")
                         + f" from {table}.")
                    ced.append({'db_id': db_id, 'question': q,
                                'schema_links': {table: sample}})

                    col_sample = rng.sample(uncovered_cols, min(6, len(uncovered_cols)))
                    for col in col_sample:
                        tmpl = rng.choice(_CED_TEMPLATES_1COL)
                        q = tmpl.format(col=col, table=table)
                        ced.append({'db_id': db_id, 'question': q,
                                    'schema_links': {table: [col]}})

                    covered_in_table = [c for c in cols
                                        if (db_id, table.lower(), c.lower()) in covered]
                    if covered_in_table and uncovered_cols:
                        c1 = rng.choice(uncovered_cols)
                        c2 = rng.choice(covered_in_table)
                        tmpl = rng.choice(_CED_TEMPLATES_2COL)
                        q = tmpl.format(col1=c1, col2=c2, table=table)
                        ced.append({'db_id': db_id, 'question': q,
                                    'schema_links': {table: [c1, c2]}})

            # ── Full coverage for never-seen validation tables ──────────────
            elif is_val_db:
                sample = rng.sample(cols, min(6, len(cols)))
                q = ("Show " + ", ".join(sample[:3])
                     + (" and " + ", ".join(sample[3:]) if len(sample) > 3 else "")
                     + f" from the {table} table.")
                ced.append({'db_id': db_id, 'question': q,
                            'schema_links': {table: sample}})

                for col in rng.sample(cols, min(5, len(cols))):
                    tmpl = rng.choice(_CED_TEMPLATES_1COL)
                    q = tmpl.format(col=col, table=table)
                    ced.append({'db_id': db_id, 'question': q,
                                'schema_links': {table: [col]}})

                if len(cols) >= 2:
                    c1, c2 = rng.sample(cols, 2)
                    tmpl = rng.choice(_CED_TEMPLATES_2COL)
                    q = tmpl.format(col1=c1, col2=c2, table=table)
                    ced.append({'db_id': db_id, 'question': q,
                                'schema_links': {table: [c1, c2]}})

        # ── 2-table FK join examples for validation databases ───────────────
        if is_val_db:
            fk_pairs = raw.get('foreign_keys', [])
            for fk_pair in fk_pairs[:8]:
                idx1, idx2 = fk_pair
                if idx1 >= len(col_info) or idx2 >= len(col_info):
                    continue
                t1_idx, _c1 = col_info[idx1]
                t2_idx, _c2 = col_info[idx2]
                if t1_idx == -1 or t2_idx == -1 or t1_idx == t2_idx:
                    continue
                t1, t2 = tnames[t1_idx], tnames[t2_idx]
                t1_other = [c for ti, c in col_info if ti == t1_idx and c != _c1]
                t2_other = [c for ti, c in col_info if ti == t2_idx and c != _c2]
                if not t1_other or not t2_other:
                    continue
                c1_extra = rng.choice(t1_other[:4])
                c2_extra = rng.choice(t2_other[:4])
                tmpl = rng.choice(_CED_FK_TEMPLATES)
                q = tmpl.format(col1=c1_extra, t1=t1, col2=c2_extra, t2=t2)
                ced.append({'db_id': db_id, 'question': q,
                            'schema_links': {t1: [c1_extra], t2: [c2_extra]}})

    rng.shuffle(ced)
    return ced

# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(train_data: list, tokenizer) -> Dataset:
    schema_fmt = CONFIG['schema_fmt']
    cache      = {}
    texts, skipped = [], 0

    for item in train_data:
        db_id = item['db_id']
        if db_id not in cache:
            try:
                raw = _load_schema_raw(db_id)
                schema, col_ann, _, _ = _build_maps(raw, schema_fmt)
                cache[db_id] = (schema, col_ann)
            except FileNotFoundError:
                skipped += 1
                continue

        schema, col_ann = cache[db_id]
        answer = item.get('schema_links') or {}
        if not isinstance(answer, dict):
            answer = {}

        user_content = f"{serialize_schema(schema, col_ann)}\n\nQuestion: {item['question']}"
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT_PKFK},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
        try:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=False)
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        except Exception:
            skipped += 1

    print(f"  Dataset built: {len(texts)} examples, {skipped} skipped", flush=True)
    return Dataset.from_dict({"text": texts})

# ── Training ──────────────────────────────────────────────────────────────────

def run_training(train_data: list) -> bool:
    log_sep("TRAINING  final adapter")
    log(f"  model={BASE_MODEL}  schema={CONFIG['schema_fmt']}")
    log(f"  r={CONFIG['lora_r']}  alpha={CONFIG['lora_alpha']}  dropout={CONFIG['lora_dropout']}")
    log(f"  lr={CONFIG['lr']}  epochs={CONFIG['epochs']}  max_length={CONFIG['max_length']}")
    log(f"  batch={CONFIG['batch_size']}  grad_accum={CONFIG['grad_accum']}")
    log(f"  train_size={len(train_data)}")
    log(f"  adapter -> {ADAPTER_DIR}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = build_dataset(train_data, tokenizer)

        print("=== Loading base model (bf16) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=CONFIG['lora_r'],
            lora_alpha=CONFIG['lora_alpha'],
            lora_dropout=CONFIG['lora_dropout'],
            target_modules=LORA_TARGETS,
            bias="none",
        )
        os.makedirs(ADAPTER_DIR, exist_ok=True)

        sft_cfg = SFTConfig(
            output_dir=ADAPTER_DIR,
            num_train_epochs=CONFIG['epochs'],
            per_device_train_batch_size=CONFIG['batch_size'],
            gradient_accumulation_steps=CONFIG['grad_accum'],
            learning_rate=CONFIG['lr'],
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=25,
            save_strategy="no",
            report_to="none",
            optim="adamw_torch_fused",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_length=CONFIG['max_length'],
            completion_only_loss=True,
            packing=False,
        )

        print(f"=== Starting training ({CONFIG['epochs']} epochs) ===", flush=True)
        trainer = SFTTrainer(
            model=model, args=sft_cfg,
            train_dataset=dataset, peft_config=lora_cfg,
        )
        trainer.train()

        print(f"=== Saving adapter -> {ADAPTER_DIR} ===", flush=True)
        trainer.model.save_pretrained(ADAPTER_DIR)
        tokenizer.save_pretrained(ADAPTER_DIR)
        log(f"  Training complete. Adapter saved -> {ADAPTER_DIR}")

        del trainer, model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        is_oom = (isinstance(e, torch.cuda.OutOfMemoryError) or
                  'out of memory' in str(e).lower())
        label = "OOM" if is_oom else "ERROR"
        log(f"  [{label}] training failed: {e}")
        log(traceback.format_exc(), also_print=False)
        torch.cuda.empty_cache()
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(42)

    with open(LOG_FILE, 'w') as f:
        f.write(f"train.py  started {_ts()}\n\n")

    log_sep("train.py  —  CED Full Coverage + pkfk")

    # ── Load augmented training set ────────────────────────────────────────
    aug_v2 = None
    for path in [TRAIN_JSON, './train.json']:
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            aug_v2 = [x for x in raw if x.get('schema_links') is not None]
            log(f"Loaded {len(aug_v2)} base training examples from {path}")
            break
    if aug_v2 is None:
        log("ERROR: no training data found."); sys.exit(1)

    # Drop zero-column examples (teach the model to always output columns)
    aug_v2_full = aug_v2
    aug_v2 = [x for x in aug_v2
              if any(cols for cols in (x.get('schema_links') or {}).values())]
    log(f"Filtered {len(aug_v2_full) - len(aug_v2)} zero-column examples "
        f"-> {len(aug_v2)} remain")

    with open(VAL_INPUT) as f:
        val_items = json.load(f)
    log(f"Validation: {len(val_items)} questions")

    # ── Generate CED ───────────────────────────────────────────────────────
    log("Generating CED (full validation DB coverage) ...")
    ced = generate_ced(aug_v2, val_items, rng)
    log(f"  CED generated: {len(ced)} examples")

    combined = aug_v2 + ced
    rng.shuffle(combined)
    log(f"Combined training set: {len(combined)} examples "
        f"(aug_v2={len(aug_v2)}, CED={len(ced)})")

    # ── Train ──────────────────────────────────────────────────────────────
    ok = run_training(combined)
    log(f"\nDone. {'OK' if ok else 'FAILED'}. Full log -> {LOG_FILE}")
    log("Run inference with:  python main.py")


if __name__ == '__main__':
    main()

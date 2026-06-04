"""
test20b.py -- test20-b experiment for RTX 4070 (12 GB VRAM, QLoRA 4-bit).

This is test20-b from test20.py adapted for 12 GB VRAM:
  - QLoRA 4-bit (BitsAndBytesConfig nf4 + double-quant + bf16 compute)
  - batch_size=1, grad_accum=8 (same effective batch as test20-a's 2×2)
  - save_strategy="no"  ← learned from test17 hang fix
  - no gradient_checkpointing_kwargs  ← learned from test17 hang fix
  - dataloader_num_workers=0  ← avoids multiprocessing deadlock

Config: Qwen3-1.7B  pkfk  r=32  alpha=32  CED-v2  3 epochs  lr=1e-4

CED v2 key fix (same as test20.py):
  test19's CED skipped 229/615 validation tables (37.2%) that were never in
  aug_v2 — including 16 NTSB tables (EVENT, AIRBAG, INTERIOR, etc.).
  test20 generates examples for ALL tables in ALL validation databases.

Other fixes:
  - Filter 79 zero-column aug_v2 examples (stop training model to output [])
  - System prompt: "Never output an empty column list []"
  - max_new_tokens=1024 (prevent truncated column lists)

Preds  → test20-b-preds.json
Log    → test20b-log.txt
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
from peft import LoraConfig, PeftModel, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────

SCHEMAS_DIR    = './schemas'
TRAIN_JSON     = './augmented_train_v2.json'
VAL_INPUT      = './validation_input.json'
LOG_FILE       = './test20b-log.txt'

LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_NEW_TOKENS = 1024

EXP = {
    'tag':          'test20-b',
    'adapter_dir':  './adapters/test20-b',
    'preds_file':   './test20-b-preds.json',
    'base_model':   'Qwen/Qwen3-1.7B',
    'is_qwen3':     True,
    'schema_fmt':   'pkfk',
    'lora_r':       32,
    'lora_alpha':   32,
    'lora_dropout': 0.05,
    'lr':           1e-4,
    'epochs':       3,
    'batch_size':   1,
    'grad_accum':   8,
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


def filter_and_dedup(links: dict, lc_tables: dict, lc_cols: dict) -> dict:
    result = {}
    for table, cols in links.items():
        canonical_t = lc_tables.get(str(table).lower())
        if canonical_t is None:
            continue
        if not isinstance(cols, list):
            result[canonical_t] = []
            continue
        col_map = lc_cols.get(canonical_t, {})
        seen, deduped = set(), []
        for c in cols:
            canonical_c = col_map.get(str(c).lower())
            if canonical_c is not None and canonical_c not in seen:
                seen.add(canonical_c)
                deduped.append(canonical_c)
        result[canonical_t] = deduped
    return result

# ── JSON parsing ──────────────────────────────────────────────────────────────

def _repair_json(text: str) -> str:
    text = re.sub(r',\s*$', '', text.rstrip())
    depth_brace = depth_bracket = 0
    in_str = esc = False
    for ch in text:
        if esc:                   esc = False; continue
        if ch == '\\' and in_str: esc = True;  continue
        if ch == '"':             in_str = not in_str; continue
        if in_str:                continue
        depth_brace   += (ch == '{') - (ch == '}')
        depth_bracket += (ch == '[') - (ch == ']')
    return text + ('"' if in_str else '') + ']' * max(0, depth_bracket) + '}' * max(0, depth_brace)


def parse_json(text: str) -> dict:
    text = text.strip()
    for candidate in [text, text[text.find('{'):text.rfind('}')+1] if '{' in text else '']:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    try:
        start = text.find('{')
        if start != -1:
            obj = json.loads(_repair_json(text[start:]))
            if isinstance(obj, dict):
                return obj
    except (json.JSONDecodeError, ValueError):
        pass
    return {}

# ── CED v2 ────────────────────────────────────────────────────────────────────

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


def generate_ced_v2(aug_v2: list, val_items: list, rng: random.Random) -> list:
    covered      = set()
    covered_tables = set()
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

            # Part A: column-extension for aug_v2-covered tables
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

                    for col in rng.sample(uncovered_cols, min(6, len(uncovered_cols))):
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

            # Part B: full coverage for never-seen validation tables
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

        # Part C: FK join examples for validation databases
        if is_val_db:
            for fk_pair in raw.get('foreign_keys', [])[:8]:
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
    cache = {}
    texts, skipped = [], 0

    for item in train_data:
        db_id = item['db_id']
        if db_id not in cache:
            try:
                raw = _load_schema_raw(db_id)
                schema, col_ann, _, _ = _build_maps(raw, EXP['schema_fmt'])
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
    tag = EXP['tag']
    log_sep(f"TRAINING  {tag}  (QLoRA 4-bit, RTX 4070)")
    log(f"  model={EXP['base_model']}  r={EXP['lora_r']}  alpha={EXP['lora_alpha']}")
    log(f"  lr={EXP['lr']}  epochs={EXP['epochs']}  max_length={EXP['max_length']}")
    log(f"  batch={EXP['batch_size']}  grad_accum={EXP['grad_accum']}")
    log(f"  train_size={len(train_data)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(EXP['base_model'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = build_dataset(train_data, tokenizer)

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        print(f"=== [{tag}] Loading base model (4-bit QLoRA) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            EXP['base_model'],
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=EXP['lora_r'],
            lora_alpha=EXP['lora_alpha'],
            lora_dropout=EXP['lora_dropout'],
            target_modules=LORA_TARGETS,
            bias="none",
        )
        os.makedirs(EXP['adapter_dir'], exist_ok=True)

        sft_cfg = SFTConfig(
            output_dir=EXP['adapter_dir'],
            num_train_epochs=EXP['epochs'],
            per_device_train_batch_size=EXP['batch_size'],
            gradient_accumulation_steps=EXP['grad_accum'],
            learning_rate=EXP['lr'],
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            # no gradient_checkpointing_kwargs — causes deadlock with 4-bit on some drivers
            logging_steps=25,
            save_strategy="no",       # avoids mid-training checkpoint hang on QLoRA
            report_to="none",
            optim="adamw_torch",      # paged_adamw_8bit also works if installed
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_length=EXP['max_length'],
            completion_only_loss=True,
            packing=False,
            dataloader_num_workers=0, # avoid multiprocessing deadlock
        )

        print(f"=== [{tag}] Starting training ({EXP['epochs']} epochs) ===", flush=True)
        trainer = SFTTrainer(
            model=model, args=sft_cfg,
            train_dataset=dataset, peft_config=lora_cfg,
        )
        trainer.train()

        print(f"=== [{tag}] Saving adapter → {EXP['adapter_dir']} ===", flush=True)
        trainer.model.save_pretrained(EXP['adapter_dir'])
        tokenizer.save_pretrained(EXP['adapter_dir'])
        log(f"  Training complete. Adapter saved → {EXP['adapter_dir']}")

        del trainer, model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        is_oom = (isinstance(e, torch.cuda.OutOfMemoryError) or
                  'out of memory' in str(e).lower())
        label = "OOM" if is_oom else "ERROR"
        log(f"  [{label}] {tag} training failed: {e}")
        log(traceback.format_exc(), also_print=False)
        torch.cuda.empty_cache()
        return False

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(val_items: list) -> bool:
    tag = EXP['tag']
    log_sep(f"INFERENCE  {tag}")

    try:
        tok_src = (EXP['adapter_dir']
                   if os.path.exists(
                       os.path.join(EXP['adapter_dir'], 'tokenizer_config.json'))
                   else EXP['base_model'])
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load in bf16 for inference (faster than 4-bit, adapter is tiny)
        print(f"=== [{tag}] Loading bf16 base + adapter for inference ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            EXP['base_model'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, EXP['adapter_dir'])
        model.eval()

        preds = []
        n = len(val_items)

        for i, item in enumerate(val_items, 1):
            qid      = item['question_id']
            db_id    = item['db_id']
            question = item['question']

            try:
                raw = _load_schema_raw(db_id)
                schema, col_ann, lc_tables, lc_cols = _build_maps(raw, EXP['schema_fmt'])
            except FileNotFoundError:
                log(f"  WARNING: schema missing for {db_id} (qid={qid})")
                preds.append({'question_id': qid, 'schema_links': {}})
                continue

            user_content = f"{serialize_schema(schema, col_ann)}\n\nQuestion: {question}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_PKFK},
                {"role": "user",   "content": user_content},
            ]

            try:
                try:
                    enc = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True,
                        return_tensors="pt", enable_thinking=False)
                except TypeError:
                    enc = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, return_tensors="pt")
            except Exception as e:
                log(f"  Tokenisation error qid={qid}: {e}")
                preds.append({'question_id': qid, 'schema_links': {}})
                continue

            input_ids = enc.input_ids if hasattr(enc, 'input_ids') else enc
            input_ids = input_ids.to(model.device)
            input_len = input_ids.shape[-1]

            with torch.no_grad():
                out = model.generate(
                    input_ids, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            raw_text  = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
            raw_links = parse_json(raw_text)

            if not raw_links:
                with torch.no_grad():
                    out2 = model.generate(
                        input_ids, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True, temperature=0.4, top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id)
                raw_links = parse_json(
                    tokenizer.decode(out2[0][input_len:], skip_special_tokens=True))

            validated = filter_and_dedup(raw_links, lc_tables, lc_cols)
            preds.append({'question_id': qid, 'schema_links': validated})

            if i % 10 == 0 or i == n:
                print(f"[{tag}] Inference {i}/{n}", flush=True)

        with open(EXP['preds_file'], 'w') as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        log(f"  Wrote {len(preds)} predictions → {EXP['preds_file']}")

        del model, base
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        is_oom = (isinstance(e, torch.cuda.OutOfMemoryError) or
                  'out of memory' in str(e).lower())
        label = "OOM" if is_oom else "ERROR"
        log(f"  [{label}] {tag} inference failed: {e}")
        log(traceback.format_exc(), also_print=False)
        torch.cuda.empty_cache()
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(42)

    with open(LOG_FILE, 'w') as f:
        f.write(f"test20b.py  started {_ts()}\n")
        f.write(f"schema=pkfk  r=32  alpha=32  hardware=RTX-4070-12GB  QLoRA-4bit\n\n")

    log_sep("test20b.py  —  QLoRA 4-bit  CED-v2  r=32  —  RTX 4070")

    # Load aug_v2
    aug_v2 = None
    for path in [TRAIN_JSON, './augmented_train_10x.json', './train.json']:
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            aug_v2 = [x for x in raw if x.get('schema_links') is not None]
            log(f"Loaded {len(aug_v2)} base training examples from {path}")
            break
    if aug_v2 is None:
        log("ERROR: no training data found."); sys.exit(1)

    # Filter zero-column examples
    aug_v2_full = aug_v2
    aug_v2 = [x for x in aug_v2
              if any(cols for cols in (x.get('schema_links') or {}).values())]
    log(f"Filtered {len(aug_v2_full) - len(aug_v2)} zero-column examples → {len(aug_v2)} remain")

    with open(VAL_INPUT) as f:
        val_items = json.load(f)
    log(f"Validation: {len(val_items)} questions")

    # Generate CED v2
    log("Generating CED v2 (full validation DB coverage) ...")
    ced = generate_ced_v2(aug_v2, val_items, rng)
    log(f"  CED v2: {len(ced)} examples")

    combined = aug_v2 + ced
    rng.shuffle(combined)
    log(f"Combined: {len(combined)} examples (aug_v2={len(aug_v2)}, CED={len(ced)})")

    # Table coverage check
    val_db_ids = set(item['db_id'] for item in val_items)
    train_tables_covered = set()
    for x in combined:
        for t in (x.get('schema_links') or {}):
            train_tables_covered.add((x['db_id'], t.lower()))
    total_val_t = covered_val_t = 0
    for db_id in val_db_ids:
        try:
            raw = _load_schema_raw(db_id)
            for t in raw['table_names_original']:
                total_val_t += 1
                if (db_id, t.lower()) in train_tables_covered:
                    covered_val_t += 1
        except FileNotFoundError:
            pass
    log(f"Val table coverage: {covered_val_t}/{total_val_t} "
        f"({100*covered_val_t/max(total_val_t,1):.1f}%)")

    # Train
    ok_train = run_training(combined)

    # Infer
    ok_infer = False
    if ok_train and os.path.isdir(EXP['adapter_dir']):
        ok_infer = run_inference(val_items)

    log_sep("SUMMARY")
    log(f"  train: {'OK' if ok_train else 'FAILED'}")
    log(f"  infer: {'OK' if ok_infer else 'FAILED'}")
    if ok_infer:
        log(f"  preds: {EXP['preds_file']}")
        log(f"\nTo evaluate:")
        log(f"  python eval.py --predictions {EXP['preds_file']} "
            f"--gold validation_gold_schema_links.json "
            f"--schemas_dir schemas/ --questions_input validation_input.json")
    log(f"\nDone. Full log → {LOG_FILE}")


if __name__ == '__main__':
    main()

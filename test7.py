"""
test7.py -- Qwen3-1.7B hyperparameter sweep on augmented_train_v2.json.

Experiments:
  qwen3_pkfk   Qwen3-1.7B  schema_sorted_pkfk  lr=2e-4  warmup=0.05  (PK/FK hints)
  qwen3_lr1e4  Qwen3-1.7B  schema_sorted        lr=1e-4  warmup=0.05  (lower LR)
  qwen3_warm   Qwen3-1.7B  schema_sorted        lr=2e-4  warmup=0.10  (longer warmup)

All use augmented_train_v2.json (same data as the current best 0.4415 run).

Adapter  → ./adapter_t7_{name}/
Preds    → ./preds_t7_{name}.json

Usage:
    python test7.py                              # run all 3 in sequence
    python test7.py qwen3_pkfk qwen3_lr1e4      # run specific experiments by name
"""

import json
import os
import subprocess
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

SCHEMAS_DIR = './schemas'
TRAIN_JSON  = './augmented_train_v2.json'
VAL_INPUT   = './validation_input.json'
VAL_GOLD    = './validation_gold_schema_links.json'

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Experiment configs ─────────────────────────────────────────────────────────
# (name, base_model, schema_format, lora_r, lora_alpha, lr, epochs, batch, grad_accum, warmup_ratio)
EXPERIMENTS = [
    ('qwen3_pkfk',  'Qwen/Qwen3-1.7B', 'schema_sorted_pkfk', 16, 32, 2e-4, 3, 2, 2, 0.05),
    ('qwen3_lr1e4', 'Qwen/Qwen3-1.7B', 'schema_sorted',      16, 32, 1e-4, 3, 2, 2, 0.05),
    ('qwen3_warm',  'Qwen/Qwen3-1.7B', 'schema_sorted',      16, 32, 2e-4, 3, 2, 2, 0.10),
]

# ── System prompts ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TYPED = (
    "You are a database assistant. "
    "Given a database schema (column types shown as col:type) and a natural language "
    "question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without the :type suffix) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)
SYSTEM_PROMPT_PKFK = (
    "You are a database assistant. "
    "Given a database schema (column types as col:type; [PK]=primary key, [FK]=foreign key) "
    "and a natural language question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without type/key suffixes) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

# ── Schema loading ─────────────────────────────────────────────────────────────

def load_schema_full(db_id, schemas_dir):
    """Return (schema, col_types, pkfk_merged)."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    table_names = s['table_names_original']
    col_info    = s['column_names_original']
    raw_types   = s.get('column_types', [])
    pk_set      = set(s.get('primary_keys', []))
    fk_set      = set()
    for pair in s.get('foreign_keys', []):
        fk_set.update(pair)

    schema = {t: [] for t in table_names}
    types  = {t: {} for t in table_names}
    pkfk   = {t: {} for t in table_names}
    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t   = table_names[tidx]
        typ = raw_types[i] if i < len(raw_types) else ''
        flag = 'PK' if i in pk_set else ('FK' if i in fk_set else '')
        schema[t].append(cname)
        types[t][cname] = typ
        pkfk[t][cname]  = f"{typ}[{flag}]" if (typ and flag) else (typ or (f"[{flag}]" if flag else ''))

    return schema, types, pkfk


# ── Schema serialization ──────────────────────────────────────────────────────

def serialize(schema, fmt, col_ann):
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        ann      = col_ann.get(table, {}) if col_ann else {}
        col_strs = [f"{c}:{ann[c]}" if ann.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
    return "Schema:\n" + "\n".join(lines)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(data, tokenizer, schema_format, schemas_dir, base_model):
    cache    = {}
    is_qwen3 = 'qwen3' in base_model.lower()
    sys_prompt = SYSTEM_PROMPT_PKFK if schema_format == 'schema_sorted_pkfk' else SYSTEM_PROMPT_TYPED

    texts, skipped = [], 0
    for item in data:
        db_id = item['db_id']
        if db_id not in cache:
            try:
                cache[db_id] = load_schema_full(db_id, schemas_dir)
            except FileNotFoundError:
                skipped += 1
                continue

        schema, col_types, pkfk_ann = cache[db_id]
        col_ann = pkfk_ann if schema_format == 'schema_sorted_pkfk' else col_types
        answer  = item.get('schema_links') or {}

        user_content = f"{serialize(schema, schema_format, col_ann)}\n\nQuestion: {item['question']}"
        messages = [
            {"role": "system",    "content": sys_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
        try:
            if is_qwen3:
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False,
                        enable_thinking=False)
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False)
            else:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        except Exception:
            skipped += 1

    print(f"  {len(texts)} formatted, {skipped} skipped")
    return Dataset.from_dict({"text": texts})


# ── Training ──────────────────────────────────────────────────────────────────

def run_experiment(name, base_model, schema_format, lora_r, lora_alpha,
                   lr, epochs, batch_size, grad_accum, warmup_ratio, train_data):
    adapter_dir = f"./adapter_t7_{name}"
    print(f"\n{'='*65}")
    print(f"  Experiment : {name}")
    print(f"  Model      : {base_model}")
    print(f"  Format     : {schema_format}")
    print(f"  LoRA       : r={lora_r} alpha={lora_alpha}")
    print(f"  LR={lr}  warmup={warmup_ratio}  epochs={epochs}")
    print(f"  Adapter    : {adapter_dir}")
    print('='*65)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset ...")
    dataset = build_dataset(train_data, tokenizer, schema_format, SCHEMAS_DIR, base_model)

    print(f"Loading base model {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGETS, bias="none",
    )
    os.makedirs(adapter_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=adapter_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch_fused",
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_length=1024,
        completion_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model, args=sft_config, train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()

    print(f"Saving adapter → {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del trainer, model
    torch.cuda.empty_cache()

    return adapter_dir


# ── Eval ──────────────────────────────────────────────────────────────────────

def run_eval(adapter_dir, base_model, schema_format, name):
    preds_file = f"./preds_t7_{name}.json"
    print(f"\nRunning eval for {name} ...")

    r1 = subprocess.run([
        sys.executable, "main.py",
        "--input",         VAL_INPUT,
        "--output",        preds_file,
        "--adapter_dir",   adapter_dir,
        "--base_model",    base_model,
        "--schema_format", schema_format,
    ], capture_output=True, text=True)
    if r1.returncode != 0:
        print("  [inference error]", r1.stderr[-500:])
        return None

    r2 = subprocess.run([
        sys.executable, "eval.py",
        "--predictions",     preds_file,
        "--gold",            VAL_GOLD,
        "--schemas_dir",     SCHEMAS_DIR,
        "--questions_input", VAL_INPUT,
    ], capture_output=True, text=True)
    print(r2.stdout)
    for line in r2.stdout.splitlines():
        if "Leaderboard Score" in line:
            try:
                return float(line.split(":")[-1].strip().split()[0])
            except ValueError:
                pass
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    names_filter = set(sys.argv[1:])
    exps = [e for e in EXPERIMENTS if not names_filter or e[0] in names_filter]
    if not exps:
        print(f"Unknown experiment(s): {names_filter}")
        print(f"Available: {[e[0] for e in EXPERIMENTS]}")
        sys.exit(1)

    print(f"Loading training data from {TRAIN_JSON} ...")
    with open(TRAIN_JSON) as f:
        raw = json.load(f)
    train_data = [x for x in raw if x.get('schema_links') is not None]
    print(f"  {len(train_data)} examples with schema_links")

    scores = {}
    for (name, base_model, schema_format, lora_r, lora_alpha,
         lr, epochs, batch_size, grad_accum, warmup_ratio) in exps:
        adapter_dir = run_experiment(
            name, base_model, schema_format, lora_r, lora_alpha,
            lr, epochs, batch_size, grad_accum, warmup_ratio, train_data)
        score = run_eval(adapter_dir, base_model, schema_format, name)
        scores[name] = score
        print(f"\n  >>> {name}: leaderboard = {score}")

    print("\n" + "="*65)
    print("SUMMARY (test7.py)")
    print("="*65)
    for n, s in scores.items():
        tag = f"{s:.4f}" if s is not None else "eval_failed"
        baseline = " (baseline: 0.4415)" if "qwen3" in n else ""
        print(f"  {n:25s}  {tag}{baseline}")

    if scores:
        best = max(scores, key=lambda k: scores[k] or 0)
        print(f"\nBest: {best}  →  copy adapter with:")
        print(f"  cp -r ./adapter_t7_{best}/* ./adapter/")


if __name__ == '__main__':
    main()

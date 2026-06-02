"""
test3.py -- 3 Qwen3-focused experiments on AWS L4 (TRL 1.5.1).
Run alongside test4.py on RTX 4070 for 6 more configs.

Analysis from test1/test2:
  - Qwen3-1.7B is the only model worth pursuing (0.4415 vs 0.36 for others)
  - Column score is the bottleneck: 0.33 now, need 0.45+ to hit LB=0.50
  - PK/FK format hurt (confused the model) — skip
  - QLoRA r=32 on Qwen2.5 barely helped — but Qwen3 has more headroom
  - Original column order hypothesis: sorted columns disrupt semantic grouping

Experiments (all Qwen3-1.7B):
  qwen3_r32       schema_sorted      r=32 α=64  3ep  test higher LoRA rank
  qwen3_orig_col  sorted_table_orig  r=16 α=32  3ep  test original col order
  qwen3_r32_orig  sorted_table_orig  r=32 α=64  3ep  combine both (most promising)

Usage:
    python test3.py
    python test3.py qwen3_r32
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
VAL_INPUT   = './validation_input.json'
VAL_GOLD    = './validation_gold_schema_links.json'
QWEN3       = 'Qwen/Qwen3-1.7B'
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# (name, schema_format, lora_r, lora_alpha, lr, epochs, batch, grad_accum)
EXPERIMENTS = [
    ('qwen3_r32',      'schema_sorted',      32, 64, 2e-4, 3, 2, 2),
    ('qwen3_orig_col', 'sorted_table_orig_col', 16, 32, 2e-4, 3, 2, 2),
    ('qwen3_r32_orig', 'sorted_table_orig_col', 32, 64, 2e-4, 3, 2, 2),
]

SYSTEM_PROMPT = (
    "You are a database assistant. "
    "Given a database schema (column types shown as col:type) and a natural language "
    "question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without the :type suffix) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

# ── Schema ────────────────────────────────────────────────────────────────────

def load_schema_full(db_id, schemas_dir):
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    table_names = s['table_names_original']
    col_info    = s['column_names_original']
    raw_types   = s.get('column_types', [])
    schema    = {t: [] for t in table_names}
    col_types = {t: {} for t in table_names}
    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t = table_names[tidx]
        schema[t].append(cname)
        col_types[t][cname] = raw_types[i] if i < len(raw_types) else ''
    return schema, col_types


def serialize(schema, fmt, col_types):
    if fmt == 'schema_sorted':
        lines = []
        for t in sorted(schema):
            cols = sorted(schema[t])
            ann  = col_types.get(t, {})
            col_strs = [f"{c}:{ann[c]}" if ann.get(c) else c for c in cols]
            lines.append(f"  {t}({', '.join(col_strs)})" if col_strs else f"  {t}")
        return "Schema:\n" + "\n".join(lines)
    if fmt == 'sorted_table_orig_col':
        lines = []
        for t in sorted(schema):
            cols = schema[t]             # original order
            ann  = col_types.get(t, {})
            col_strs = [f"{c}:{ann[c]}" if ann.get(c) else c for c in cols]
            lines.append(f"  {t}({', '.join(col_strs)})" if col_strs else f"  {t}")
        return "Schema:\n" + "\n".join(lines)
    lines = [f"  {t}({', '.join(c)})" if c else f"  {t}" for t, c in schema.items()]
    return "Schema:\n" + "\n".join(lines)


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset(data, tokenizer, schema_format, schemas_dir):
    cache = {}
    texts, skipped = [], 0
    for item in data:
        db_id = item['db_id']
        if db_id not in cache:
            try:
                cache[db_id] = load_schema_full(db_id, schemas_dir)
            except FileNotFoundError:
                skipped += 1; continue
        schema, col_types = cache[db_id]
        answer = item.get('schema_links') or {}
        if not isinstance(answer, dict):
            answer = {}
        schema_text  = serialize(schema, schema_format, col_types)
        user_content = f"{schema_text}\n\nQuestion: {item['question']}"
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
        try:
            # Qwen3: disable thinking so output is pure JSON
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
    print(f"  {len(texts)} formatted, {skipped} skipped")
    return Dataset.from_dict({"text": texts})


# ── Training ──────────────────────────────────────────────────────────────────

def run_experiment(name, schema_format, lora_r, lora_alpha, lr, epochs,
                   batch_size, grad_accum, train_data):
    adapter_dir = f"./adapter_t3_{name}"
    print(f"\n{'='*65}")
    print(f"  Experiment : {name}")
    print(f"  Model      : {QWEN3}")
    print(f"  Format     : {schema_format}")
    print(f"  LoRA       : r={lora_r} α={lora_alpha}  lr={lr}  epochs={epochs}")
    print(f"  Adapter    : {adapter_dir}")
    print('='*65)

    tokenizer = AutoTokenizer.from_pretrained(QWEN3)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset ...")
    dataset = build_dataset(train_data, tokenizer, schema_format, SCHEMAS_DIR)

    print(f"Loading {QWEN3} ...")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN3, device_map="auto", torch_dtype=torch.bfloat16)
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
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_length=1024,
        completion_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model, args=sft_config,
        train_dataset=dataset, peft_config=lora_config,
    )
    trainer.train()

    print(f"Saving → {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    del trainer, model
    torch.cuda.empty_cache()
    return adapter_dir


# ── Eval ──────────────────────────────────────────────────────────────────────

def run_eval(adapter_dir, schema_format, name):
    preds_file = f"./preds_t3_{name}.json"
    print(f"\nEvaluating {name} ...")
    r1 = subprocess.run([
        sys.executable, "main.py",
        "--input", VAL_INPUT, "--output", preds_file,
        "--adapter_dir", adapter_dir, "--schema_format", schema_format,
        "--base_model", QWEN3,
    ], capture_output=True, text=True)
    if r1.returncode != 0:
        print("  [inference error]", r1.stderr[-400:]); return None
    r2 = subprocess.run([
        sys.executable, "eval.py",
        "--predictions", preds_file, "--gold", VAL_GOLD,
        "--schemas_dir", SCHEMAS_DIR, "--questions_input", VAL_INPUT,
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
        print(f"Available: {[e[0] for e in EXPERIMENTS]}"); sys.exit(1)

    for path in ['./augmented_train_v2.json', './augmented_train_10x.json', './train.json']:
        if os.path.exists(path):
            print(f"Loading {path} ...")
            with open(path) as f:
                raw = json.load(f)
            train_data = [x for x in raw if x.get('schema_links') is not None]
            print(f"  {len(train_data)} examples"); break
    else:
        print("ERROR: no training data"); sys.exit(1)

    scores = {}
    for (name, fmt, r, alpha, lr, ep, bs, ga) in exps:
        adapter_dir = run_experiment(name, fmt, r, alpha, lr, ep, bs, ga, train_data)
        scores[name] = run_eval(adapter_dir, fmt, name)
        print(f"\n  >>> {name}: {scores[name]}")

    print("\n" + "="*65)
    print("SUMMARY (test3.py)  —  prev best Qwen3: 0.4415")
    print("="*65)
    for n, s in scores.items():
        bar = f"{s:.4f}" if s else "failed"
        marker = " ← NEW BEST" if s and s > 0.4415 else ""
        print(f"  {n:<25}  {bar}{marker}")
    best = max(scores, key=lambda k: scores[k] or 0)
    print(f"\nBest: {best}  →  cp -r ./adapter_t3_{best}/* ./adapter/")


if __name__ == '__main__':
    main()

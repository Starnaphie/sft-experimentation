"""
test4.py -- 3 Qwen3-focused experiments on RTX 4070 12 GB (TRL 0.21.0+).
Run alongside test3.py on AWS L4 for 6 more configs.

Analysis from test1/test2:
  - Qwen3-1.7B is the only model worth pursuing (0.4415)
  - Column score bottleneck: 0.33 now, need 0.45+ for LB=0.50
  - QLoRA r=32 on Qwen2.5 barely helped — but Qwen3 has more headroom
  - Sorted_table_orig_col hypothesis: original col order preserves col semantics

Experiments (all Qwen3-1.7B):
  qwen3_qlora_r32   schema_sorted      QLoRA 4-bit r=32  3ep  (more rank, less VRAM)
  qwen3_ep5         schema_sorted      LoRA r=16         5ep  (more training)
  qwen3_r32_ep5     sorted_table_orig  QLoRA 4-bit r=32  5ep  (best combo, most ambitious)

Usage:
    python test4.py
    python test4.py qwen3_ep5

NOTE: run `python augment.py --target 3000 --out augmented_train_v2.json` first
      if that file doesn't exist yet.
"""

import json
import os
import subprocess
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

SCHEMAS_DIR  = './schemas'
VAL_INPUT    = './validation_input.json'
VAL_GOLD     = './validation_gold_schema_links.json'
QWEN3        = 'Qwen/Qwen3-1.7B'
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# (name, schema_format, lora_r, lora_alpha, lr, epochs, batch, grad_accum, use_qlora)
EXPERIMENTS = [
    ('qwen3_qlora_r32', 'schema_sorted',         32, 64, 2e-4, 3, 2, 4, True),
    ('qwen3_ep5',       'schema_sorted',          16, 32, 2e-4, 5, 1, 8, False),
    ('qwen3_r32_ep5',   'sorted_table_orig_col',  32, 64, 2e-4, 5, 1, 8, True),
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
                   batch_size, grad_accum, use_qlora, train_data):
    adapter_dir = f"./adapter_t4_{name}"
    print(f"\n{'='*65}")
    print(f"  Experiment : {name}")
    print(f"  Model      : {QWEN3}")
    print(f"  Format     : {schema_format}")
    print(f"  QLoRA      : {use_qlora}  r={lora_r} α={lora_alpha}")
    print(f"  lr={lr}  epochs={epochs}  batch={batch_size}  grad_accum={grad_accum}")
    print(f"  Adapter    : {adapter_dir}")
    print('='*65)

    tokenizer = AutoTokenizer.from_pretrained(QWEN3)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset ...")
    dataset = build_dataset(train_data, tokenizer, schema_format, SCHEMAS_DIR)

    if use_qlora:
        print("Loading Qwen3 in 4-bit (QLoRA) ...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            QWEN3, quantization_config=bnb_cfg, device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        print("Loading Qwen3 in BF16 ...")
        model = AutoModelForCausalLM.from_pretrained(
            QWEN3, device_map="auto", torch_dtype=torch.bfloat16)
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGETS, bias="none",
    )
    os.makedirs(adapter_dir, exist_ok=True)

    sft_kwargs = dict(
        output_dir=adapter_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch_fused",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_length=1024,
        packing=False,
    )
    # completion_only_loss and gradient_checkpointing_kwargs depend on TRL version
    try:
        SFTConfig(output_dir="/tmp", completion_only_loss=True)
        sft_kwargs["completion_only_loss"] = True
    except TypeError:
        pass
    try:
        SFTConfig(output_dir="/tmp", gradient_checkpointing_kwargs={"use_reentrant": False})
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    except TypeError:
        pass

    sft_config = SFTConfig(**sft_kwargs)
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
    preds_file = f"./preds_t4_{name}.json"
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
    for (name, fmt, r, alpha, lr, ep, bs, ga, qlora) in exps:
        adapter_dir = run_experiment(name, fmt, r, alpha, lr, ep, bs, ga, qlora, train_data)
        scores[name] = run_eval(adapter_dir, fmt, name)
        print(f"\n  >>> {name}: {scores[name]}")

    print("\n" + "="*65)
    print("SUMMARY (test4.py)  —  prev best Qwen3: 0.4415")
    print("="*65)
    for n, s in scores.items():
        bar = f"{s:.4f}" if s else "failed"
        marker = " ← NEW BEST" if s and s > 0.4415 else ""
        print(f"  {n:<25}  {bar}{marker}")
    best = max(scores, key=lambda k: scores[k] or 0)
    print(f"\nBest: {best}  →  cp -r ./adapter_t4_{best}/* ./adapter/")
    print("\nTo eval from AWS after scp:")
    print(f"  python main.py --input validation_input.json --output preds.json \\")
    print(f"      --adapter_dir ./adapter_t4_{best} --schema_format <fmt> --base_model {QWEN3}")


if __name__ == '__main__':
    main()

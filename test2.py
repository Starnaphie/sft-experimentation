"""
test2.py -- 3 sequential SFT experiments on RTX 4070 12 GB (TRL 0.21.0).
Run while AWS L4 runs test1.py — together they cover 6 distinct configs.

Experiments:
  qlora_r32     Qwen2.5-1.5B   schema_sorted  QLoRA 4-bit r=32   (higher capacity, less VRAM)
  smollm        SmolLM2-1.7B   schema_sorted  LoRA  r=16         (different architecture)
  qwen05_5ep    Qwen2.5-0.5B   schema_sorted  LoRA  r=32  5ep    (small model, more epochs)

Usage:
    python test2.py               # run all 3 in sequence
    python test2.py qlora_r32     # run single experiment by name

Adapter  → ./adapter_t2_{name}/
Preds    → ./preds_t2_{name}.json

Requirements (already in your cse234 env):
    torch, transformers, peft, trl==0.21.0, datasets, bitsandbytes, sqlglot
"""

import json
import os
import subprocess
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

SCHEMAS_DIR = './schemas'
TRAIN_JSON  = './augmented_train_v2.json'
VAL_INPUT   = './validation_input.json'
VAL_GOLD    = './validation_gold_schema_links.json'

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Experiment configs ─────────────────────────────────────────────────────────
# (name, base_model, schema_format, lora_r, lora_alpha, lr, epochs, batch, grad_accum, use_qlora)
EXPERIMENTS = [
    ('qlora_r32',  'Qwen/Qwen2.5-1.5B-Instruct',          'schema_sorted', 32, 64, 2e-4, 3, 2, 4, True),
    ('smollm',     'HuggingFaceTB/SmolLM2-1.7B-Instruct',  'schema_sorted', 16, 32, 2e-4, 3, 1, 8, False),
    ('qwen05_5ep', 'Qwen/Qwen2.5-0.5B-Instruct',           'schema_sorted', 32, 64, 2e-4, 5, 2, 4, False),
]

SYSTEM_PROMPT_TYPED = (
    "You are a database assistant. "
    "Given a database schema (column types shown as col:type) and a natural language "
    "question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without the :type suffix) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

# ── Schema loading ─────────────────────────────────────────────────────────────

def load_schema_full(db_id, schemas_dir):
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    table_names = s['table_names_original']
    col_info    = s['column_names_original']
    raw_types   = s.get('column_types', [])
    schema      = {t: [] for t in table_names}
    col_types   = {t: {} for t in table_names}
    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t = table_names[tidx]
        schema[t].append(cname)
        col_types[t][cname] = raw_types[i] if i < len(raw_types) else ''
    return schema, col_types


def serialize(schema, col_types):
    """schema_sorted serialization (tables + cols A→Z, with type annotations)."""
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        t_types  = col_types.get(table, {})
        col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
    return "Schema:\n" + "\n".join(lines)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(data, tokenizer, schemas_dir):
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
        schema_text  = serialize(schema, col_types)
        user_content = f"{schema_text}\n\nQuestion: {item['question']}"
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT_TYPED},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        except Exception:
            skipped += 1
    print(f"  {len(texts)} formatted, {skipped} skipped")
    return Dataset.from_dict({"text": texts})


# ── Training ──────────────────────────────────────────────────────────────────

def run_experiment(name, base_model, schema_format, lora_r, lora_alpha,
                   lr, epochs, batch_size, grad_accum, use_qlora, train_data):
    adapter_dir = f"./adapter_t2_{name}"
    print(f"\n{'='*65}")
    print(f"  Experiment : {name}")
    print(f"  Model      : {base_model}")
    print(f"  QLoRA      : {use_qlora}  LoRA r={lora_r} alpha={lora_alpha}")
    print(f"  lr={lr}  epochs={epochs}  batch={batch_size}  grad_accum={grad_accum}")
    print(f"  Adapter    : {adapter_dir}")
    print('='*65)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset ...")
    dataset = build_dataset(train_data, tokenizer, SCHEMAS_DIR)

    if use_qlora:
        print("Loading model in 4-bit (QLoRA) ...")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb_cfg, device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"Loading model in BF16 ...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", torch_dtype=torch.bfloat16)
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=LORA_TARGETS, bias="none",
    )
    os.makedirs(adapter_dir, exist_ok=True)

    training_args = TrainingArguments(
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
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # TRL 0.21.0 renamed `tokenizer` → `processing_class`.
    # DataCollatorForCompletionOnlyLM was also moved; skip gracefully if absent.
    response_template = "<|im_start|>assistant\n"
    collator = None
    for mod_path in ["trl", "trl.trainer.utils", "trl.data_utils"]:
        try:
            import importlib
            m = importlib.import_module(mod_path)
            DCRL = getattr(m, "DataCollatorForCompletionOnlyLM")
            collator = DCRL(response_template=response_template, tokenizer=tokenizer)
            print(f"  Using DataCollatorForCompletionOnlyLM from {mod_path}")
            break
        except Exception:
            pass
    if collator is None:
        print("  [info] DataCollatorForCompletionOnlyLM not found; training on all tokens")

    trainer_kwargs = dict(
        model=model,
        processing_class=tokenizer,   # TRL 0.21.0+ uses processing_class
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=1024,
    )
    if collator is not None:
        trainer_kwargs["data_collator"] = collator

    # Older TRL might still use 'tokenizer'; fall back if processing_class is rejected
    try:
        trainer = SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs["tokenizer"] = trainer_kwargs.pop("processing_class")
        trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    print(f"Saving adapter → {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del trainer, model
    torch.cuda.empty_cache()
    return adapter_dir


# ── Eval ──────────────────────────────────────────────────────────────────────

def run_eval(adapter_dir, schema_format, name):
    preds_file = f"./preds_t2_{name}.json"
    print(f"\nRunning eval for {name} ...")
    r1 = subprocess.run([
        sys.executable, "main.py",
        "--input",  VAL_INPUT,
        "--output", preds_file,
        "--adapter_dir",   adapter_dir,
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
        print(f"Unknown: {names_filter}. Available: {[e[0] for e in EXPERIMENTS]}")
        sys.exit(1)

    for path in [TRAIN_JSON, './augmented_train_10x.json', './train.json']:
        if os.path.exists(path):
            print(f"Loading training data from {path} ...")
            with open(path) as f:
                raw = json.load(f)
            train_data = [x for x in raw if x.get('schema_links') is not None]
            print(f"  {len(train_data)} examples")
            break
    else:
        print("ERROR: no training data found"); sys.exit(1)

    scores = {}
    for (name, base_model, schema_format, lora_r, lora_alpha,
         lr, epochs, batch_size, grad_accum, use_qlora) in exps:
        adapter_dir = run_experiment(
            name, base_model, schema_format, lora_r, lora_alpha,
            lr, epochs, batch_size, grad_accum, use_qlora, train_data)
        score = run_eval(adapter_dir, schema_format, name)
        scores[name] = score
        print(f"\n  >>> {name}: leaderboard = {score}")

    print("\n" + "="*65)
    print("SUMMARY (test2.py)")
    print("="*65)
    for n, s in scores.items():
        tag = f"{s:.4f}" if s is not None else "eval_failed"
        print(f"  {n:25s}  {tag}")

    best = max(scores, key=lambda k: scores[k] or 0)
    print(f"\nBest: {best}  →  copy adapter with:")
    print(f"  cp -r ./adapter_t2_{best}/* ./adapter/")
    print("\nNOTE: if evaluating from the AWS machine, scp the adapter folder first:")
    print(f"  scp -r <user>@<4070-host>:~/sft-experimentation/adapter_t2_{best} ./")


if __name__ == '__main__':
    main()

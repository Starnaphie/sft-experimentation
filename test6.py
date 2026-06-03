"""
test6.py -- Qwen2.5-1.5B trained on augmented_train_v3.json, schema_sorted_pkfk format.

Mirrors test1.py:v2_pkfk but uses the new v3 augmented data (5 000 examples).

Adapter  → ./adapter_t6_v3_pkfk/
Preds    → ./preds_t6_v3_pkfk.json

Usage:
    python test6.py
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

SCHEMAS_DIR  = './schemas'
TRAIN_JSON   = './augmented_train_v4.json'
VAL_INPUT    = './validation_input.json'
VAL_GOLD     = './validation_gold_schema_links.json'

BASE_MODEL    = 'Qwen/Qwen2.5-1.5B-Instruct'
SCHEMA_FORMAT = 'schema_sorted_pkfk'
LORA_R        = 16
LORA_ALPHA    = 32
LR            = 2e-4
EPOCHS        = 3
BATCH_SIZE    = 2
GRAD_ACCUM    = 2
ADAPTER_DIR   = './adapter_t6_v4_pkfk'
PREDS_FILE    = './preds_t6_v4_pkfk.json'
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj"]

SYSTEM_PROMPT = (
    "You are a database assistant. "
    "Given a database schema (column types as col:type; [PK]=primary key, [FK]=foreign key) "
    "and a natural language question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without type/key suffixes) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)


def load_schema_full(db_id, schemas_dir):
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
    pkfk   = {t: {} for t in table_names}
    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t   = table_names[tidx]
        typ = raw_types[i] if i < len(raw_types) else ''
        flag = 'PK' if i in pk_set else ('FK' if i in fk_set else '')
        schema[t].append(cname)
        pkfk[t][cname] = f"{typ}[{flag}]" if (typ and flag) else (typ or (f"[{flag}]" if flag else ''))
    return schema, pkfk


def serialize(schema, pkfk):
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        ann      = pkfk.get(table, {})
        col_strs = [f"{c}:{ann[c]}" if ann.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
    return "Schema:\n" + "\n".join(lines)


def build_dataset(data, tokenizer, schemas_dir):
    cache = {}
    texts, skipped = [], 0
    for item in data:
        db_id = item['db_id']
        if db_id not in cache:
            try:
                cache[db_id] = load_schema_full(db_id, schemas_dir)
            except FileNotFoundError:
                skipped += 1
                continue

        schema, pkfk = cache[db_id]
        answer = item.get('schema_links') or {}

        user_content = f"{serialize(schema, pkfk)}\n\nQuestion: {item['question']}"
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
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


def train():
    print(f"\n{'='*65}")
    print(f"  test6: Qwen2.5-1.5B + augmented_train_v4 (clean, ~3000 examples)")
    print(f"  Format: schema_sorted_pkfk (PK/FK hints)")
    print(f"  LoRA r={LORA_R} alpha={LORA_ALPHA}  lr={LR}  epochs={EPOCHS}")
    print(f"  Adapter → {ADAPTER_DIR}")
    print('='*65)

    print(f"Loading training data from {TRAIN_JSON} ...")
    with open(TRAIN_JSON) as f:
        raw = json.load(f)
    train_data = [x for x in raw if x.get('schema_links') is not None]
    print(f"  {len(train_data)} examples")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset ...")
    dataset = build_dataset(train_data, tokenizer, SCHEMAS_DIR)

    print(f"Loading base model {BASE_MODEL} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=LORA_TARGETS, bias="none",
    )
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=ADAPTER_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
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
        model=model, args=sft_config, train_dataset=dataset,
        peft_config=lora_config,
    )
    trainer.train()

    print(f"Saving adapter → {ADAPTER_DIR}")
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    del trainer, model
    torch.cuda.empty_cache()


def evaluate():
    print(f"\nRunning inference on validation set ...")
    r1 = subprocess.run([
        sys.executable, "main.py",
        "--input",         VAL_INPUT,
        "--output",        PREDS_FILE,
        "--adapter_dir",   ADAPTER_DIR,
        "--base_model",    BASE_MODEL,
        "--schema_format", SCHEMA_FORMAT,
    ], capture_output=True, text=True)
    if r1.returncode != 0:
        print("  [inference error]", r1.stderr[-800:])
        return

    print(f"Evaluating {PREDS_FILE} ...")
    r2 = subprocess.run([
        sys.executable, "eval.py",
        "--predictions",     PREDS_FILE,
        "--gold",            VAL_GOLD,
        "--schemas_dir",     SCHEMAS_DIR,
        "--questions_input", VAL_INPUT,
    ], capture_output=True, text=True)
    print(r2.stdout)


if __name__ == '__main__':
    train()
    evaluate()

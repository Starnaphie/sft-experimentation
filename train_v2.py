"""
train_v2.py -- SFT training with HuggingFace TRL SFTTrainer (TRL >= 1.5, no RapidFireAI).

Usage:
    python train_v2.py                              # augmented_train_v2.json → ./adapter/
    python train_v2.py --train_json train.json      # original 301 examples
    python train_v2.py --schema_format sorted_table_orig_col
    python train_v2.py --epochs 5 --lr 5e-5

Activate environment first:
    conda activate cse234

Adapter is saved to ./adapter/ and a format-tagged copy to ./adapter_v2_<format>/.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE_MODEL  = 'Qwen/Qwen2.5-1.5B-Instruct'
TRAIN_JSON  = './augmented_train_v2.json'
SCHEMAS_DIR = './schemas'
ADAPTER_DIR = './adapter'

NUM_EPOCHS  = 3
LR          = 2e-4
BATCH_SIZE  = 2
GRAD_ACCUM  = 2
MAX_SEQ_LEN = 1024

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TYPED = (
    "You are a database assistant. "
    "Given a database schema (column types shown as col:type) and a natural language "
    "question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without the :type suffix) from the schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

SYSTEM_PROMPT_PLAIN = (
    "You are a database assistant. "
    "Given a database schema and a natural language question, output the schema links "
    "as a JSON object: {\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names that appear in the given schema. "
    "Include only the tables and columns needed to answer the question. "
    "Output valid JSON only, with no extra text."
)

# ── Schema loading ─────────────────────────────────────────────────────────────

def load_schema(db_id: str, schemas_dir: str):
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        s = json.load(f)
    schema   = {t: [] for t in s['table_names_original']}
    col_types = {t: {} for t in s['table_names_original']}
    raw_types = s.get('column_types', [])
    for i, (tidx, cname) in enumerate(s['column_names_original']):
        if tidx == -1:
            continue
        t = s['table_names_original'][tidx]
        schema[t].append(cname)
        if i < len(raw_types):
            col_types[t][cname] = raw_types[i]
    return schema, col_types


# ── Schema serialization ──────────────────────────────────────────────────────

def serialize_schema(schema: dict, fmt: str, col_types: dict = None) -> str:
    if fmt == 'schema_sorted':
        lines = []
        for table in sorted(schema.keys()):
            cols     = sorted(schema[table])
            t_types  = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    if fmt == 'sorted_table_orig_col':
        lines = []
        for table in sorted(schema.keys()):
            cols     = schema[table]          # original order
            t_types  = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    if fmt == 'typed':
        lines = []
        for table, cols in schema.items():
            t_types  = col_types.get(table, {}) if col_types else {}
            col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
            lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    lines = [f"  {t}({', '.join(c)})" if c else f"  {t}" for t, c in schema.items()]
    return "Schema:\n" + "\n".join(lines)


def get_system_prompt(fmt: str) -> str:
    return SYSTEM_PROMPT_TYPED if fmt in ('schema_sorted', 'sorted_table_orig_col', 'typed') \
           else SYSTEM_PROMPT_PLAIN


# ── Dataset preparation ───────────────────────────────────────────────────────

def build_dataset(data: list, tokenizer, schemas_dir: str, schema_format: str) -> Dataset:
    schema_cache = {}
    sys_prompt   = get_system_prompt(schema_format)

    texts, skipped = [], 0
    for item in data:
        db_id = item['db_id']
        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = load_schema(db_id, schemas_dir)
            except FileNotFoundError:
                skipped += 1
                continue

        schema, col_types = schema_cache[db_id]
        answer = item.get('schema_links') or {}
        if not isinstance(answer, dict):
            answer = {}

        schema_text  = serialize_schema(schema, schema_format, col_types)
        user_content = f"{schema_text}\n\nQuestion: {item['question']}"

        messages = [
            {"role": "system",    "content": sys_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        except Exception:
            skipped += 1

    print(f"  {len(texts)} examples formatted, {skipped} skipped")
    return Dataset.from_dict({"text": texts})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json',    default=TRAIN_JSON)
    ap.add_argument('--schemas_dir',   default=SCHEMAS_DIR)
    ap.add_argument('--adapter_dir',   default=ADAPTER_DIR)
    ap.add_argument('--base_model',    default=BASE_MODEL)
    ap.add_argument('--schema_format', default='schema_sorted',
                    choices=['schema_sorted', 'sorted_table_orig_col', 'typed', 'compact'])
    ap.add_argument('--epochs',        type=int,   default=NUM_EPOCHS)
    ap.add_argument('--lr',            type=float, default=LR)
    ap.add_argument('--batch_size',    type=int,   default=BATCH_SIZE)
    ap.add_argument('--grad_accum',    type=int,   default=GRAD_ACCUM)
    ap.add_argument('--max_seq_len',   type=int,   default=MAX_SEQ_LEN)
    args = ap.parse_args()

    # Fall back to original data if augmented file doesn't exist
    if not os.path.exists(args.train_json):
        fallback = './train.json'
        print(f"[warn] {args.train_json} not found — falling back to {fallback}")
        args.train_json = fallback

    print(f"Loading {args.train_json} ...")
    with open(args.train_json) as f:
        raw_data = json.load(f)
    data = [x for x in raw_data if x.get('schema_links') is not None]
    print(f"  {len(data)} / {len(raw_data)} examples have schema_links")

    print(f"Loading tokenizer from {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building dataset ...")
    dataset = build_dataset(data, tokenizer, args.schemas_dir, args.schema_format)

    print(f"Loading base model {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )

    fmt_adapter_dir = f"./adapter_v2_{args.schema_format}"
    os.makedirs(args.adapter_dir, exist_ok=True)
    os.makedirs(fmt_adapter_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=args.adapter_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        optim="adamw_torch_fused",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_length=args.max_seq_len,
        completion_only_loss=True,   # train only on the assistant (JSON) tokens
        packing=False,
    )

    print(f"\nTraining  epochs={args.epochs}  lr={args.lr}  "
          f"batch={args.batch_size}  grad_accum={args.grad_accum}")
    print(f"  schema_format       : {args.schema_format}")
    print(f"  adapter_dir         : {args.adapter_dir}")
    print(f"  examples            : {len(dataset)}")
    print(f"  completion_only_loss: True")
    print()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=None,   # dataset already has "text" column
    )
    trainer.train()

    print(f"\nSaving adapter → {args.adapter_dir}")
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    print(f"Saving copy  → {fmt_adapter_dir}")
    trainer.model.save_pretrained(fmt_adapter_dir)
    tokenizer.save_pretrained(fmt_adapter_dir)

    print("\nDone. Evaluate with:")
    print(f"  python main.py --input validation_input.json --output predictions.json \\")
    print(f"      --adapter_dir {args.adapter_dir} --schema_format {args.schema_format}")
    print(f"  python eval.py --predictions predictions.json \\")
    print(f"      --gold validation_gold_schema_links.json \\")
    print(f"      --schemas_dir schemas/ --questions_input validation_input.json")


if __name__ == '__main__':
    main()

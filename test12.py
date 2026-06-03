"""
test12.py -- High-capacity LoRA experiments: larger r/alpha vs more epochs.

Experiments (sequential, same base otherwise as Method 3):
  test12-a  Qwen3-1.7B  schema_sorted  aug_v2  LoRA r=64  alpha=128  lr=2e-4  3 epochs
  test12-b  Qwen3-1.7B  schema_sorted  aug_v2  LoRA r=32  alpha=64   lr=2e-4  5 epochs

Baseline to beat: Leaderboard=0.4415  Table=0.5538  Column=0.3292
                  (Method 3: r=16, alpha=32, 3 epochs)

After training each adapter, the script:
  1. Runs in-process inference on validation_input.json
  2. Applies post-processing: filters hallucinated tables/columns against
     actual Spider-format schemas in ./schemas/
  3. Saves predictions to test12-a-preds.json / test12-b-preds.json
  4. Logs start/end timestamps + config summary to test12-log.txt

OOM errors are caught and logged; the script continues to the next experiment.

DO NOT execute this script directly — run it manually:
    conda run -n cse234 python test12.py
    # or inside tmux:
    nohup conda run -n cse234 python test12.py > test12-output.log 2>&1 &
"""

import datetime
import json
import os
import re
import sys
import traceback

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────

SCHEMAS_DIR = './schemas'
TRAIN_JSON  = './augmented_train_v2.json'
VAL_INPUT   = './validation_input.json'
LOG_FILE    = './test12-log.txt'

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
BASE_MODEL   = 'Qwen/Qwen3-1.7B'
SCHEMA_FMT   = 'schema_sorted'
MAX_NEW_TOKENS = 512

# ── Experiment definitions ────────────────────────────────────────────────────
# (tag, adapter_dir, preds_file, lora_r, lora_alpha, lr, epochs)

EXPERIMENTS = [
    {
        'tag':         'test12-a',
        'adapter_dir': './adapter_test12_a',
        'preds_file':  './test12-a-preds.json',
        'lora_r':      64,
        'lora_alpha':  128,
        'lr':          2e-4,
        'epochs':      3,
        'batch_size':  2,
        'grad_accum':  2,
        'max_length':  1024,
    },
    {
        'tag':         'test12-b',
        'adapter_dir': './adapter_test12_b',
        'preds_file':  './test12-b-preds.json',
        'lora_r':      32,
        'lora_alpha':  64,
        'lr':          2e-4,
        'epochs':      5,
        'batch_size':  2,
        'grad_accum':  2,
        'max_length':  1024,
    },
]

# ── System prompt (matches Method 3 / schema_sorted) ─────────────────────────

SYSTEM_PROMPT = (
    "You are a database assistant. "
    "Given a database schema (column types shown as col:type) and a natural language "
    "question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without the :type suffix) from the schema. "
    "Include only the tables and columns needed to answer the question. "
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


def log_section(title: str):
    sep = '=' * 65
    log(sep)
    log(f"  {title}")
    log(sep)


# ── Schema helpers ────────────────────────────────────────────────────────────

def load_schema_raw(db_id: str, schemas_dir: str) -> dict:
    """Return raw Spider schema dict (cached by caller)."""
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    with open(os.path.join(schemas_dir, fname)) as f:
        return json.load(f)


def build_schema_maps(raw: dict):
    """Return (schema, col_types, lc_tables, lc_cols).

    schema:    {table: [col, ...]}          original-case
    col_types: {table: {col: type_str}}
    lc_tables: {table_lower: table_orig}    for case-insensitive lookup
    lc_cols:   {table_orig: {col_lower: col_orig}}
    """
    table_names = raw['table_names_original']
    col_info    = raw['column_names_original']
    col_types   = raw.get('column_types', [])

    schema    = {t: [] for t in table_names}
    ctypes    = {t: {} for t in table_names}

    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t = table_names[tidx]
        schema[t].append(cname)
        ctypes[t][cname] = col_types[i] if i < len(col_types) else ''

    lc_tables = {t.lower(): t for t in schema}
    lc_cols   = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}
    return schema, ctypes, lc_tables, lc_cols


def serialize_schema(schema: dict, col_types: dict) -> str:
    """schema_sorted: tables A→Z, columns A→Z, col:type annotations."""
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        t_types  = col_types.get(table, {})
        col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
    return "Schema:\n" + "\n".join(lines)


def filter_against_schema(links: dict, lc_tables: dict, lc_cols: dict) -> dict:
    """Drop hallucinated tables/columns; restore canonical casing.

    Case-insensitive on both table and column names.
    Tables with empty column lists (COUNT(*) queries) are kept.
    """
    result = {}
    for table, cols in links.items():
        canonical_t = lc_tables.get(str(table).lower())
        if canonical_t is None:
            continue                      # hallucinated table — drop
        if not isinstance(cols, list):
            result[canonical_t] = []
            continue
        col_map = lc_cols.get(canonical_t, {})
        result[canonical_t] = [
            col_map[str(c).lower()]
            for c in cols if str(c).lower() in col_map
        ]
    return result


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _repair_json(text: str) -> str:
    """Close unclosed brackets to recover from truncated output."""
    text = re.sub(r',\s*$', '', text.rstrip())
    depth_brace = depth_bracket = 0
    in_str = esc = False
    for c in text:
        if esc:
            esc = False; continue
        if c == '\\' and in_str:
            esc = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        depth_brace   += (c == '{') - (c == '}')
        depth_bracket += (c == '[') - (c == ']')
    suffix = ('"' if in_str else '') + (']' * max(0, depth_bracket)) + ('}' * max(0, depth_brace))
    return text + suffix


def parse_json(text: str) -> dict:
    """Three-stage parser: direct → extract outermost {} → repair truncation."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    if start == -1:
        return {}
    end = text.rfind('}')
    if end > start:
        try:
            obj = json.loads(text[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    try:
        obj = json.loads(_repair_json(text[start:]))
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return {}


# ── Training dataset builder ──────────────────────────────────────────────────

def build_dataset(train_data: list, tokenizer, schemas_dir: str) -> Dataset:
    schema_cache = {}
    texts, skipped = [], 0

    for item in train_data:
        db_id = item['db_id']
        if db_id not in schema_cache:
            try:
                raw = load_schema_raw(db_id, schemas_dir)
                schema, col_types, _, _ = build_schema_maps(raw)
                schema_cache[db_id] = (schema, col_types)
            except FileNotFoundError:
                skipped += 1
                continue

        schema, col_types = schema_cache[db_id]
        answer = item.get('schema_links') or {}
        if not isinstance(answer, dict):
            answer = {}

        schema_text  = serialize_schema(schema, col_types)
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

    print(f"  Dataset: {len(texts)} formatted, {skipped} skipped", flush=True)
    return Dataset.from_dict({"text": texts})


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(exp: dict, train_data: list) -> bool:
    """Train a LoRA adapter. Returns True on success, False on OOM/error."""
    tag         = exp['tag']
    adapter_dir = exp['adapter_dir']

    log_section(f"TRAINING  {tag}")
    log(f"  base_model : {BASE_MODEL}")
    log(f"  schema_fmt : {SCHEMA_FMT}")
    log(f"  train_data : {TRAIN_JSON}  ({len(train_data)} examples)")
    log(f"  LoRA       : r={exp['lora_r']}  alpha={exp['lora_alpha']}")
    log(f"  lr={exp['lr']}  epochs={exp['epochs']}  batch={exp['batch_size']}  "
        f"grad_accum={exp['grad_accum']}  max_length={exp['max_length']}")
    log(f"  adapter_dir: {adapter_dir}")

    try:
        print(f"\n[{tag}] Loading tokenizer ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[{tag}] Building dataset ...", flush=True)
        dataset = build_dataset(train_data, tokenizer, SCHEMAS_DIR)

        print(f"[{tag}] Loading base model {BASE_MODEL} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16)
        model.enable_input_require_grads()

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=exp['lora_r'],
            lora_alpha=exp['lora_alpha'],
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
            bias="none",
        )
        os.makedirs(adapter_dir, exist_ok=True)

        sft_config = SFTConfig(
            output_dir=adapter_dir,
            num_train_epochs=exp['epochs'],
            per_device_train_batch_size=exp['batch_size'],
            gradient_accumulation_steps=exp['grad_accum'],
            learning_rate=exp['lr'],
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
            max_length=exp['max_length'],
            completion_only_loss=True,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model, args=sft_config, train_dataset=dataset,
            peft_config=lora_config,
        )

        print(f"[{tag}] Starting training ...", flush=True)
        trainer.train()

        print(f"[{tag}] Saving adapter → {adapter_dir}", flush=True)
        trainer.model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        log(f"  Training complete. Adapter saved → {adapter_dir}")

        del trainer, model
        torch.cuda.empty_cache()
        return True

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            log(f"  OOM ERROR during training {tag}: {e}")
            log(f"  Skipping {tag} and continuing.")
        else:
            log(f"  RuntimeError during training {tag}: {e}")
            log(traceback.format_exc())
        torch.cuda.empty_cache()
        return False

    except Exception as e:
        log(f"  Unexpected error during training {tag}: {e}")
        log(traceback.format_exc())
        torch.cuda.empty_cache()
        return False


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(exp: dict, val_items: list) -> bool:
    """Load adapter and predict schema links for all validation items.

    Applies post-processing schema validation before saving predictions.
    Returns True on success, False on OOM/error.
    """
    tag         = exp['tag']
    adapter_dir = exp['adapter_dir']
    preds_file  = exp['preds_file']

    log_section(f"INFERENCE  {tag}")
    log(f"  adapter_dir : {adapter_dir}")
    log(f"  preds_file  : {preds_file}")

    # Pre-load all schemas needed for post-processing
    schema_map_cache = {}
    for item in val_items:
        db_id = item['db_id']
        if db_id not in schema_map_cache:
            try:
                raw = load_schema_raw(db_id, SCHEMAS_DIR)
                _, _, lc_tables, lc_cols = build_schema_maps(raw)
                schema_map_cache[db_id] = (lc_tables, lc_cols)
            except FileNotFoundError:
                log(f"  WARNING: schema not found for {db_id}")
                schema_map_cache[db_id] = ({}, {})

    try:
        print(f"\n[{tag}] Loading tokenizer from {adapter_dir} ...", flush=True)
        tok_src = adapter_dir if os.path.exists(
            os.path.join(adapter_dir, 'tokenizer_config.json')) else BASE_MODEL
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[{tag}] Loading base model ...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16)

        print(f"[{tag}] Loading LoRA adapter from {adapter_dir} ...", flush=True)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()

        preds = []
        n = len(val_items)

        for i, item in enumerate(val_items, 1):
            db_id    = item['db_id']
            question = item['question']
            qid      = item['question_id']

            # Build schema text for the prompt
            try:
                raw = load_schema_raw(db_id, SCHEMAS_DIR)
                schema, col_types, lc_tables, lc_cols = build_schema_maps(raw)
            except FileNotFoundError:
                log(f"  WARNING: schema missing for {db_id}, question_id={qid}")
                preds.append({'question_id': qid, 'schema_links': {}})
                continue

            schema_text  = serialize_schema(schema, col_types)
            user_content = f"{schema_text}\n\nQuestion: {question}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ]

            # Tokenise
            try:
                try:
                    ids = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True,
                        return_tensors="pt", enable_thinking=False)
                except TypeError:
                    ids = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True,
                        return_tensors="pt")
            except Exception as e:
                log(f"  Tokenisation error for qid={qid}: {e}")
                preds.append({'question_id': qid, 'schema_links': {}})
                continue

            input_ids = ids.input_ids if hasattr(ids, 'input_ids') else ids
            input_ids = input_ids.to(model.device)
            input_len = input_ids.shape[-1]

            # Greedy decode
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
            raw_links = parse_json(raw_text)

            # Fallback: temperature sampling if greedy produced nothing
            if not raw_links:
                with torch.no_grad():
                    out2 = model.generate(
                        input_ids,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True,
                        temperature=0.4,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                raw_text  = tokenizer.decode(out2[0][input_len:], skip_special_tokens=True)
                raw_links = parse_json(raw_text)

            # Post-processing: filter against actual schema (case-insensitive)
            validated = filter_against_schema(raw_links, lc_tables, lc_cols)
            preds.append({'question_id': qid, 'schema_links': validated})

            if i % 10 == 0 or i == n:
                print(f"[{tag}] Inference {i}/{n}", flush=True)

        # Save predictions
        with open(preds_file, 'w') as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        log(f"  Wrote {len(preds)} predictions → {preds_file}")

        del model, base
        torch.cuda.empty_cache()
        return True

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            log(f"  OOM ERROR during inference {tag}: {e}")
        else:
            log(f"  RuntimeError during inference {tag}: {e}")
            log(traceback.format_exc())
        torch.cuda.empty_cache()
        return False

    except Exception as e:
        log(f"  Unexpected error during inference {tag}: {e}")
        log(traceback.format_exc())
        torch.cuda.empty_cache()
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Initialise log
    with open(LOG_FILE, 'w') as f:
        f.write(f"test12.py  started at {_ts()}\n")
        f.write(f"base_model={BASE_MODEL}  schema_fmt={SCHEMA_FMT}\n")
        f.write(f"train_data={TRAIN_JSON}\n\n")

    log(f"=== test12.py starting — {len(EXPERIMENTS)} experiments ===")

    # Load training data once
    train_data = None
    for path in [TRAIN_JSON, './augmented_train_10x.json', './train.json']:
        if os.path.exists(path):
            log(f"Loading training data from {path} ...")
            with open(path) as f:
                raw = json.load(f)
            train_data = [x for x in raw if x.get('schema_links') is not None]
            log(f"  {len(train_data)} examples with schema_links")
            break

    if train_data is None:
        log("ERROR: no training data found. Exiting.")
        sys.exit(1)

    # Load validation items once
    with open(VAL_INPUT) as f:
        val_items = json.load(f)
    log(f"Validation set: {len(val_items)} questions")

    # ── Phase 1: train all experiments ────────────────────────────────────────
    log("\n" + "="*65)
    log("  PHASE 1: TRAINING")
    log("="*65)

    trained = {}
    for exp in EXPERIMENTS:
        tag = exp['tag']
        log(f"\nStarting training for {tag} at {_ts()}")
        success = run_training(exp, train_data)
        trained[tag] = success
        status = "SUCCESS" if success else "FAILED/SKIPPED"
        log(f"Training {tag} → {status}  (finished at {_ts()})")

    # ── Phase 2: inference on all successfully trained adapters ───────────────
    log("\n" + "="*65)
    log("  PHASE 2: INFERENCE + POST-PROCESSING")
    log("="*65)

    inferred = {}
    for exp in EXPERIMENTS:
        tag = exp['tag']
        if not trained.get(tag):
            log(f"\nSkipping inference for {tag} (training did not complete).")
            inferred[tag] = False
            continue
        if not os.path.isdir(exp['adapter_dir']):
            log(f"\nSkipping inference for {tag} (adapter_dir not found).")
            inferred[tag] = False
            continue

        log(f"\nStarting inference for {tag} at {_ts()}")
        success = run_inference(exp, val_items)
        inferred[tag] = success
        status = "SUCCESS" if success else "FAILED"
        log(f"Inference {tag} → {status}  (finished at {_ts()})")

    # ── Summary ───────────────────────────────────────────────────────────────
    log_section("SUMMARY")
    log(f"{'experiment':<20}  {'training':<10}  {'inference':<10}  preds_file")
    for exp in EXPERIMENTS:
        tag   = exp['tag']
        tr    = 'OK' if trained.get(tag)   else 'FAILED'
        inf   = 'OK' if inferred.get(tag)  else 'FAILED'
        preds = exp['preds_file'] if inferred.get(tag) else '—'
        log(f"  {tag:<18}  {tr:<10}  {inf:<10}  {preds}")

    log("")
    log("To evaluate predictions, run:")
    for exp in EXPERIMENTS:
        if inferred.get(exp['tag']):
            log(f"  conda run -n cse234 python eval.py "
                f"--predictions {exp['preds_file']} "
                f"--gold validation_gold_schema_links.json "
                f"--schemas_dir schemas/ "
                f"--questions_input validation_input.json")
    log(f"\nAll done at {_ts()}. Full log → {LOG_FILE}")


if __name__ == '__main__':
    main()

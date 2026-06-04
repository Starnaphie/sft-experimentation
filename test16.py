"""
test16.py -- Full bf16 LoRA experiments on RTX PRO 6000 (~48 GB VRAM).

Goal: counter the severe overfitting seen in Method 3 (loss→0.05, acc→99%
by epoch 0.5).  Three regularisation strategies tested back-to-back:

  test16-a  lr=5e-5  dropout=0.10  epochs=3  len=1024  (slow lr + mild dropout)
  test16-b  lr=2e-4  dropout=0.10  epochs=2  len=1024  (early stop before overfit)
  test16-c  lr=5e-5  dropout=0.20  epochs=3  len=2048  (strong reg + long ctx)

All share Method-3 base:
  Qwen3-1.7B  schema_sorted  aug_v2  r=16  alpha=32
  batch_size=2  grad_accum=2  completion_only_loss=True  enable_thinking=False
  No quantisation (full bf16 LoRA, ~48 GB VRAM available)

Adapters  → ./adapters/test16-{a,b,c}/
Preds     → test16-{a,b,c}-preds.json
Log       → test16-log.txt

Usage (inside tmux on DSMLP RTX PRO 6000):
    conda run -n cse234 python test16.py
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

SCHEMAS_DIR    = './schemas'
TRAIN_JSON     = './augmented_train_v2.json'
VAL_INPUT      = './validation_input.json'
LOG_FILE       = './test16-log.txt'

LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]
BASE_MODEL     = 'Qwen/Qwen3-1.7B'
MAX_NEW_TOKENS = 512

# ── Experiment definitions ────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        'tag':          'test16-a',
        'adapter_dir':  './adapters/test16-a',
        'preds_file':   './test16-a-preds.json',
        'lora_r':       16,
        'lora_alpha':   32,
        'lora_dropout': 0.1,
        'lr':           5e-5,
        'epochs':       3,
        'batch_size':   2,
        'grad_accum':   2,
        'max_length':   1024,
        'hypothesis':   'slow lr + mild dropout prevents early overfit',
    },
    {
        'tag':          'test16-b',
        'adapter_dir':  './adapters/test16-b',
        'preds_file':   './test16-b-preds.json',
        'lora_r':       16,
        'lora_alpha':   32,
        'lora_dropout': 0.1,
        'lr':           2e-4,
        'epochs':       2,
        'batch_size':   2,
        'grad_accum':   2,
        'max_length':   1024,
        'hypothesis':   'stop early before overfit kicks in ~epoch 0.5',
    },
    {
        'tag':          'test16-c',
        'adapter_dir':  './adapters/test16-c',
        'preds_file':   './test16-c-preds.json',
        'lora_r':       16,
        'lora_alpha':   32,
        'lora_dropout': 0.2,
        'lr':           5e-5,
        'epochs':       3,
        'batch_size':   2,
        'grad_accum':   2,
        'max_length':   2048,
        'hypothesis':   'strong regularisation + longer context combined',
    },
]

# ── System prompt (schema_sorted / Method 3) ──────────────────────────────────

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


def _build_maps(raw: dict):
    """Return (schema, col_types, lc_tables, lc_cols)."""
    tnames      = raw['table_names_original']
    col_info    = raw['column_names_original']
    ctypes_list = raw.get('column_types', [])

    schema = {t: [] for t in tnames}
    ctypes = {t: {} for t in tnames}

    for i, (tidx, cname) in enumerate(col_info):
        if tidx == -1:
            continue
        t = tnames[tidx]
        schema[t].append(cname)
        ctypes[t][cname] = ctypes_list[i] if i < len(ctypes_list) else ''

    lc_tables = {t.lower(): t for t in schema}
    lc_cols   = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}
    return schema, ctypes, lc_tables, lc_cols


def serialize_schema(schema: dict, col_types: dict) -> str:
    """schema_sorted: tables A→Z, columns A→Z, with col:type annotations."""
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        t_types  = col_types.get(table, {})
        col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
    return "Schema:\n" + "\n".join(lines)


def filter_and_dedup(links: dict, lc_tables: dict, lc_cols: dict) -> dict:
    """Drop hallucinated tables/columns, restore canonical casing, deduplicate columns."""
    result = {}
    for table, cols in links.items():
        canonical_t = lc_tables.get(str(table).lower())
        if canonical_t is None:
            continue
        if not isinstance(cols, list):
            result[canonical_t] = []
            continue
        col_map = lc_cols.get(canonical_t, {})
        seen = set()
        deduped = []
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
        if esc:                esc = False; continue
        if ch == '\\' and in_str: esc = True; continue
        if ch == '"':          in_str = not in_str; continue
        if in_str:             continue
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

# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(train_data: list, tokenizer) -> Dataset:
    cache = {}
    texts, skipped = [], 0

    for item in train_data:
        db_id = item['db_id']
        if db_id not in cache:
            try:
                raw = _load_schema_raw(db_id)
                schema, ctypes, _, _ = _build_maps(raw)
                cache[db_id] = (schema, ctypes)
            except FileNotFoundError:
                skipped += 1
                continue

        schema, ctypes = cache[db_id]
        answer = item.get('schema_links') or {}
        if not isinstance(answer, dict):
            answer = {}

        user_content = f"{serialize_schema(schema, ctypes)}\n\nQuestion: {item['question']}"
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

    print(f"  Dataset built: {len(texts)} examples, {skipped} skipped", flush=True)
    return Dataset.from_dict({"text": texts})

# ── Training ──────────────────────────────────────────────────────────────────

def run_training(exp: dict, train_data: list) -> bool:
    tag = exp['tag']
    log_sep(f"TRAINING  {tag}")
    log(f"  Full bf16 LoRA (no quantisation)  r={exp['lora_r']}  alpha={exp['lora_alpha']}")
    log(f"  dropout={exp['lora_dropout']}  lr={exp['lr']}  epochs={exp['epochs']}")
    log(f"  batch={exp['batch_size']}  grad_accum={exp['grad_accum']}  "
        f"max_length={exp['max_length']}")
    log(f"  hypothesis: {exp['hypothesis']}")
    log(f"  adapter → {exp['adapter_dir']}")

    try:
        print(f"\n=== Starting {tag} ===", flush=True)
        print(f"=== [{tag}] Loading tokenizer ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"=== [{tag}] Building dataset ===", flush=True)
        dataset = build_dataset(train_data, tokenizer)

        print(f"=== [{tag}] Loading base model (bf16, no quant) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.config.use_cache = False

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=exp['lora_r'],
            lora_alpha=exp['lora_alpha'],
            lora_dropout=exp['lora_dropout'],
            target_modules=LORA_TARGETS,
            bias="none",
        )
        os.makedirs(exp['adapter_dir'], exist_ok=True)

        sft_cfg = SFTConfig(
            output_dir=exp['adapter_dir'],
            num_train_epochs=exp['epochs'],
            per_device_train_batch_size=exp['batch_size'],
            gradient_accumulation_steps=exp['grad_accum'],
            learning_rate=exp['lr'],
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=25,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            optim="adamw_torch",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_length=exp['max_length'],
            completion_only_loss=True,
            packing=False,
        )

        print(f"=== [{tag}] Starting training ({exp['epochs']} epochs) ===", flush=True)
        trainer = SFTTrainer(
            model=model, args=sft_cfg,
            train_dataset=dataset, peft_config=lora_cfg,
        )
        trainer.train()

        print(f"=== [{tag}] Saving adapter → {exp['adapter_dir']} ===", flush=True)
        trainer.model.save_pretrained(exp['adapter_dir'])
        tokenizer.save_pretrained(exp['adapter_dir'])
        log(f"  Training complete. Adapter saved → {exp['adapter_dir']}")

        del trainer, model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        is_oom = (isinstance(e, torch.cuda.OutOfMemoryError) or
                  ('out of memory' in str(e).lower()))
        label = "OOM" if is_oom else "ERROR"
        log(f"  [{label}] {tag} training failed: {e}")
        log(traceback.format_exc(), also_print=False)
        torch.cuda.empty_cache()
        return False

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(exp: dict, val_items: list) -> bool:
    """Load bf16 base + LoRA adapter and predict schema links."""
    tag = exp['tag']
    log_sep(f"INFERENCE  {tag}")
    log(f"  adapter : {exp['adapter_dir']}")
    log(f"  output  : {exp['preds_file']}")

    try:
        tok_src = (exp['adapter_dir']
                   if os.path.exists(os.path.join(exp['adapter_dir'], 'tokenizer_config.json'))
                   else BASE_MODEL)
        print(f"\n=== [{tag}] Loading tokenizer ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"=== [{tag}] Loading bf16 base model + adapter ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, exp['adapter_dir'])
        model.eval()

        preds = []
        n = len(val_items)

        for i, item in enumerate(val_items, 1):
            qid      = item['question_id']
            db_id    = item['db_id']
            question = item['question']

            try:
                raw = _load_schema_raw(db_id)
                schema, ctypes, lc_tables, lc_cols = _build_maps(raw)
            except FileNotFoundError:
                log(f"  WARNING: schema missing for {db_id} (qid={qid})")
                preds.append({'question_id': qid, 'schema_links': {}})
                continue

            user_content = f"{serialize_schema(schema, ctypes)}\n\nQuestion: {question}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
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

            # Greedy first; fallback to sampling if empty
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

        with open(exp['preds_file'], 'w') as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)
        log(f"  Wrote {len(preds)} predictions → {exp['preds_file']}")

        del model, base
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        is_oom = (isinstance(e, torch.cuda.OutOfMemoryError) or
                  ('out of memory' in str(e).lower()))
        label = "OOM" if is_oom else "ERROR"
        log(f"  [{label}] {tag} inference failed: {e}")
        log(traceback.format_exc(), also_print=False)
        torch.cuda.empty_cache()
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(LOG_FILE, 'w') as f:
        f.write(f"test16.py  started {_ts()}\n")
        f.write(f"base_model={BASE_MODEL}  quant=none (full bf16 LoRA)\n")
        f.write(f"schema=schema_sorted  train={TRAIN_JSON}  hardware=RTX-PRO-6000-48GB\n\n")
        for exp in EXPERIMENTS:
            f.write(
                f"  {exp['tag']}: lr={exp['lr']}  dropout={exp['lora_dropout']}  "
                f"epochs={exp['epochs']}  max_length={exp['max_length']}\n"
                f"    hypothesis: {exp['hypothesis']}\n"
            )
        f.write('\n')

    log_sep("test16.py  —  RTX PRO 6000 48GB  —  full bf16 LoRA  —  regularisation sweep")

    # Load training data
    train_data = None
    for path in [TRAIN_JSON, './augmented_train_10x.json', './train.json']:
        if os.path.exists(path):
            with open(path) as f:
                raw = json.load(f)
            train_data = [x for x in raw if x.get('schema_links') is not None]
            log(f"Loaded {len(train_data)} training examples from {path}")
            break
    if train_data is None:
        log("ERROR: no training data found."); sys.exit(1)

    with open(VAL_INPUT) as f:
        val_items = json.load(f)
    log(f"Validation: {len(val_items)} questions")

    # ── Phase 1: train all experiments sequentially ───────────────────────────
    log_sep("PHASE 1 — TRAINING")
    trained = {}
    for exp in EXPERIMENTS:
        log(f"\n>>> Starting training: {exp['tag']}  at {_ts()}")
        ok = run_training(exp, train_data)
        trained[exp['tag']] = ok
        log(f">>> Finished training: {exp['tag']}  {'OK' if ok else 'FAILED'}  at {_ts()}")

    # ── Phase 2: inference for each successfully trained experiment ───────────
    log_sep("PHASE 2 — INFERENCE + POST-PROCESSING")
    inferred = {}
    for exp in EXPERIMENTS:
        tag = exp['tag']
        if not trained.get(tag) or not os.path.isdir(exp['adapter_dir']):
            log(f"Skipping inference for {tag} (training did not complete).")
            inferred[tag] = False
            continue
        log(f"\n>>> Starting inference: {tag}  at {_ts()}")
        ok = run_inference(exp, val_items)
        inferred[tag] = ok
        log(f">>> Finished inference: {tag}  {'OK' if ok else 'FAILED'}  at {_ts()}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log_sep("SUMMARY")
    log(f"  {'experiment':<14}  train   infer   preds_file")
    for exp in EXPERIMENTS:
        tag = exp['tag']
        tr  = 'OK    ' if trained.get(tag)  else 'FAILED'
        inf = 'OK    ' if inferred.get(tag) else 'FAILED'
        pf  = exp['preds_file'] if inferred.get(tag) else '—'
        log(f"  {tag:<14}  {tr}  {inf}  {pf}")

    log("\nTo evaluate:")
    for exp in EXPERIMENTS:
        if inferred.get(exp['tag']):
            log(f"  python eval.py --predictions {exp['preds_file']} "
                f"--gold validation_gold_schema_links.json "
                f"--schemas_dir schemas/ --questions_input validation_input.json")
    log(f"\nDone. Full log → {LOG_FILE}")


if __name__ == '__main__':
    main()

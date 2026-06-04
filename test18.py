"""
test18.py -- Full bf16 LoRA on RTX PRO 6000 (~48 GB VRAM).

Analysis of all previous experiments identified two proven independent gains
that have never been combined:

  Factor A (from test1-3, LB=0.4415 — best ever):
    Qwen2.5-1.5B-Instruct + pkfk schema format  →  Table Score 0.5538
    PK/FK annotations give the model structural context about join keys.

  Factor B (from test11, LB=0.4278 — best Qwen3):
    alpha=16 (== r, no effective scaling) + max_length=2048  →  Column Score 0.3629
    Longer context fits schemas without truncation; no LoRA scaling sharpens column recall.

Root cause of test16 failure: max_length=1024 truncates many completion targets
→ model learns to produce empty or wrong outputs (33-40 empty/101 in test16-a/b).
test16-c (only experiment with len=2048) scored 0.3798 despite strong dropout.

Experiments (sequential):

  test18-a  Qwen2.5-1.5B-Instruct + pkfk + alpha=16 + max_length=2048
            = test1-3's winning model & format  +  test11's alpha/context
            Hypothesis: best table recall from pkfk × best column score from alpha=16

  test18-b  Qwen3-1.7B + pkfk + alpha=16 + max_length=2048
            = test7-1's pkfk Qwen3 config  +  test11's alpha/context
            Hypothesis: Qwen3's stronger base + pkfk + alpha=16 combo

All shared settings (Method 3 base + test11 improvements):
  r=16  lora_dropout=0.05  lr=2e-4  3 epochs
  batch_size=2  grad_accum=2  completion_only_loss=True  enable_thinking=False

Adapters → ./adapters/test18-{a,b}/
Preds    → test18-{a,b}-preds.json
Log      → test18-log.txt

Usage (inside tmux on DSMLP RTX PRO 6000):
    conda run -n cse234 python test18.py
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
LOG_FILE       = './test18-log.txt'

LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_NEW_TOKENS = 512

# ── Experiment definitions ────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        'tag':          'test18-a',
        'adapter_dir':  './adapters/test18-a',
        'preds_file':   './test18-a-preds.json',
        'base_model':   'Qwen/Qwen2.5-1.5B-Instruct',
        'is_qwen3':     False,
        'schema_fmt':   'pkfk',
        'lora_r':       16,
        'lora_alpha':   16,      # alpha == r: no effective scaling (test11 finding)
        'lora_dropout': 0.05,
        'lr':           2e-4,
        'epochs':       3,
        'batch_size':   2,
        'grad_accum':   2,
        'max_length':   2048,
        'hypothesis':   'test1-3 winning model+pkfk × test11 alpha=16+len=2048',
    },
    {
        'tag':          'test18-b',
        'adapter_dir':  './adapters/test18-b',
        'preds_file':   './test18-b-preds.json',
        'base_model':   'Qwen/Qwen3-1.7B',
        'is_qwen3':     True,
        'schema_fmt':   'pkfk',
        'lora_r':       16,
        'lora_alpha':   16,
        'lora_dropout': 0.05,
        'lr':           2e-4,
        'epochs':       3,
        'batch_size':   2,
        'grad_accum':   2,
        'max_length':   2048,
        'hypothesis':   'test7-1 Qwen3+pkfk × test11 alpha=16+len=2048',
    },
]

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_PKFK = (
    "You are a database assistant. "
    "Given a database schema (column types as col:type; [PK]=primary key, [FK]=foreign key) "
    "and a natural language question, output the schema links as a JSON object: "
    "{\"TableName\": [\"col1\", \"col2\"]}. "
    "Use ONLY table and column names (without type/key suffixes) from the schema. "
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


def _build_maps(raw: dict, schema_fmt: str):
    """
    Return (schema, col_ann, lc_tables, lc_cols).
    col_ann: {table: {col: annotation_str}}
      - 'pkfk' fmt: annotation is 'type[PK]' / 'type[FK]' / 'type' / '[PK]' / ''
      - other fmts: annotation is just the column type string
    """
    tnames      = raw['table_names_original']
    col_info    = raw['column_names_original']
    ctypes_list = raw.get('column_types', [])

    pk_set = set(raw.get('primary_keys', []))
    fk_set = set()
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
    """Tables sorted A→Z, columns sorted A→Z, with annotation suffix."""
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        ann      = col_ann.get(table, {})
        col_strs = [f"{c}:{ann[c]}" if ann.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")
    return "Schema:\n" + "\n".join(lines)


def filter_and_dedup(links: dict, lc_tables: dict, lc_cols: dict) -> dict:
    """Drop hallucinated tables/columns, restore canonical casing, deduplicate."""
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

# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(train_data: list, tokenizer, exp: dict) -> Dataset:
    schema_fmt = exp['schema_fmt']
    is_qwen3   = exp['is_qwen3']
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

    print(f"  Dataset built: {len(texts)} examples, {skipped} skipped", flush=True)
    return Dataset.from_dict({"text": texts})

# ── Training ──────────────────────────────────────────────────────────────────

def run_training(exp: dict, train_data: list) -> bool:
    tag = exp['tag']
    log_sep(f"TRAINING  {tag}")
    log(f"  model={exp['base_model']}  schema={exp['schema_fmt']}")
    log(f"  r={exp['lora_r']}  alpha={exp['lora_alpha']}  dropout={exp['lora_dropout']}")
    log(f"  lr={exp['lr']}  epochs={exp['epochs']}  max_length={exp['max_length']}")
    log(f"  batch={exp['batch_size']}  grad_accum={exp['grad_accum']}")
    log(f"  hypothesis: {exp['hypothesis']}")
    log(f"  adapter → {exp['adapter_dir']}")

    try:
        print(f"\n=== Starting {tag} ===", flush=True)
        print(f"=== [{tag}] Loading tokenizer ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(exp['base_model'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"=== [{tag}] Building dataset ===", flush=True)
        dataset = build_dataset(train_data, tokenizer, exp)

        print(f"=== [{tag}] Loading base model (bf16, no quant) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            exp['base_model'],
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
            optim="adamw_torch_fused",
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
    tag        = exp['tag']
    is_qwen3   = exp['is_qwen3']
    schema_fmt = exp['schema_fmt']
    log_sep(f"INFERENCE  {tag}")
    log(f"  adapter : {exp['adapter_dir']}")
    log(f"  output  : {exp['preds_file']}")

    try:
        tok_src = (exp['adapter_dir']
                   if os.path.exists(os.path.join(exp['adapter_dir'], 'tokenizer_config.json'))
                   else exp['base_model'])
        print(f"\n=== [{tag}] Loading tokenizer ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"=== [{tag}] Loading bf16 base model + adapter ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            exp['base_model'],
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
                schema, col_ann, lc_tables, lc_cols = _build_maps(raw, schema_fmt)
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
                if is_qwen3:
                    try:
                        enc = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True,
                            return_tensors="pt", enable_thinking=False)
                    except TypeError:
                        enc = tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, return_tensors="pt")
                else:
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
        f.write(f"test18.py  started {_ts()}\n")
        f.write(f"schema=pkfk  train={TRAIN_JSON}  hardware=RTX-PRO-6000-48GB\n\n")
        for exp in EXPERIMENTS:
            f.write(
                f"  {exp['tag']}: {exp['base_model']}  alpha={exp['lora_alpha']}  "
                f"lr={exp['lr']}  max_length={exp['max_length']}\n"
                f"    hypothesis: {exp['hypothesis']}\n"
            )
        f.write('\n')

    log_sep("test18.py  —  RTX PRO 6000 48GB  —  pkfk × alpha=16 × len=2048")

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

    # ── Phase 2: inference ────────────────────────────────────────────────────
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

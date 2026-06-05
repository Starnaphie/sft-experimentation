"""
test25.py -- CED-v2 base (test20-a recipe) + recall post-processing.
              Target: RTX 4070 (12 GB), OOM-safe.  Same method as test24.

WHY test21-23 stalled BELOW test20-a (0.4990):  it was NOT empties.
  Per-question diff of test20-a vs test23 shows the CED-v3 + 32 hardcoded
  examples OVERFIT to NYSED and shifted the model from a high-recall
  "over-predict" posture (test20-a: avg 2.42 tables / 5.0 cols, table-recall
  superset 51/101) to a low-recall "calibrated" posture (test23: 1.63 tables /
  3.21 cols, 43/101). Under the set metric (P+R+F1)/3, over-prediction has a
  partial-credit floor; a precise-but-wrong pick scores 0. So CED-v3 WON NYSED
  but LOST ATBI / Klamath / NTSB / PacificIslands / SBO — net negative.
  Conclusion: keep test20-a's CED-v2 high-recall base; drop the overfit hardcoding.

WHAT test25 adds on top of the proven test20-a recipe:
  Recall post-processing on the validated model output (simulated on
  test20-a-preds: 0.4990 -> 0.5152):
    - column augment: for each predicted table add up to 3 schema columns whose
      name tokens overlap the question (also fills `{table: []}` empties).
    - table augment: add an un-predicted table whose NAME token-matches the
      question AND has >=2 keyword-matching columns.

OOM-safety for the 12 GB RTX 4070:
  - batch=1, grad_accum=4 (effective batch 4 — identical to test20-a's 2x2).
  - max_length=2048 (proven; activation peak ~5-6 GB for the 1.7B LM head).
  - attn_implementation="sdpa" (O(seq) memory, not O(seq^2)).
  - optim="adamw_torch" (fused kernel may be unavailable on the 4070).
  - gradient_checkpointing without use_reentrant kwarg (avoids a 4070 deadlock).
  - dataloader_num_workers=0; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

Recipe (single experiment, same data/seed as test20-a):
  test25  Qwen3-1.7B  pkfk  r=16  alpha=32  CED-v2  4 epochs  lr=2e-4  len=2048

Adapter → ./adapters/test25/   Preds → test25-preds.json   Log → test25-log.txt
"""

import datetime
import json
import os

# Reduce fragmentation OOM on the 12 GB 4070 (and avoid the MIG NVML path).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
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
LOG_FILE       = './test25-log.txt'

LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_NEW_TOKENS = 1024   # up from 512 — prevent truncated column lists

# Self-consistency: number of stochastic decodes in addition to the greedy one.
# 2 keeps inference ~3x cost (fine on the 4070); greedy + 2 samples = 3 candidates.
ENSEMBLE_SAMPLES = 2

# ── Experiment definitions ────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        'tag':          'test25',
        'adapter_dir':  './adapters/test25',
        'preds_file':   './test25-preds.json',
        'base_model':   'Qwen/Qwen3-1.7B',
        'is_qwen3':     True,
        'schema_fmt':   'pkfk',
        'lora_r':       16,
        'lora_alpha':   32,
        'lora_dropout': 0.05,
        'lr':           2e-4,
        'epochs':       4,
        'batch_size':   1,      # 12GB 4070: batch=1 keeps activations small
        'grad_accum':   4,      # effective batch 4 — matches test20-a's 2x2
        'max_length':   2048,
        'hypothesis':   'Reproduce test20-a (CED-v2) + recall post-processing → ~0.515, OOM-safe on 4070',
    },
]

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

# ── Recall post-processing ────────────────────────────────────────────────────
# The leaderboard metric is set-based (P+R+F1)/3. Over-prediction has a partial-
# credit floor (recall); a precise-but-wrong pick scores 0. test20-a already
# over-predicts TABLES (recall 0.64) but UNDER-predicts COLUMNS (recall 0.44).
# Augmenting predicted tables with question-keyword columns, and adding a clearly
# name-matching missed table, lifts recall. Simulated on test20-a-preds: +0.0162
# (0.4990 → 0.5152). col_topk=3 + table-add(name match & >=2 col matches) was best.

_PP_STOP = {
    'the','and','for','that','with','from','show','list','count','number','each',
    'all','have','has','are','was','were','which','what','how','many','their',
    'they','this','these','those','where','when','who','whom','than','into','per',
    'by','of','in','on','to','a','an','is','be','as','at','or','display','give',
    'find','get','average','total','sum','max','min','most','least','value',
    'values','name','names',
}

def _pp_tok(s: str) -> set:
    return {w for w in re.findall(r'[a-z0-9]+', str(s).lower())
            if len(w) >= 3 and w not in _PP_STOP}

def postprocess_links(validated: dict, question: str, schema: dict,
                      col_topk: int = 3, tbl_min_colmatch: int = 2) -> dict:
    """Recall-boosting post-processing. `schema` is {table: [orig-case cols]}."""
    qt = _pp_tok(question)
    out = {t: list(cols) for t, cols in validated.items()}

    # 1) column augment each predicted table (also fills `{table: []}` empties)
    for t in list(out):
        real = schema.get(t, [])
        cur  = set(out[t])
        add  = [c for c in real if c not in cur and (_pp_tok(c) & qt)]
        out[t] = (out[t] + add[:col_topk]) if out[t] else (add[:col_topk] or real[:2])

    # 2) table augment: add a missed table whose NAME matches the question AND
    #    has >=tbl_min_colmatch keyword-matching columns (conservative).
    if qt:
        for t, real in schema.items():
            if t in out:
                continue
            if _pp_tok(t) & qt:
                colmatch = [c for c in real if _pp_tok(c) & qt]
                if len(colmatch) >= tbl_min_colmatch:
                    out[t] = colmatch[:col_topk]

    return out


def ensemble_links(candidates: list, col_vote: int = 2) -> dict:
    """Self-consistency merge of several decoded `{table: [cols]}` candidates.

    Tables: UNION across candidates (maximise recall — the metric's strong lever).
    Columns per table: keep a column if it appears in >= col_vote candidates that
    predicted that table; if none clear the bar, fall back to the union (so we
    never emit an empty list). Candidates are already canonical (filter_and_dedup).
    """
    from collections import Counter
    candidates = [c for c in candidates if c]
    if not candidates:
        return {}
    tables = set()
    for c in candidates:
        tables.update(c.keys())
    out = {}
    for t in tables:
        col_counts = Counter()
        union_cols = []
        for c in candidates:
            if t in c:
                for col in c[t]:
                    col_counts[col] += 1
                    if col not in union_cols:
                        union_cols.append(col)
        voted = [col for col in union_cols if col_counts[col] >= col_vote]
        out[t] = voted if voted else union_cols
    return out

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

# ── Coverage Extension Data v2 ────────────────────────────────────────────────

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
    """
    CED v2 — key fix over test19's generate_ced():

    test19 skipped tables not in aug_v2 to "avoid confusing the model."
    This left 229 validation tables (37.2%) completely uncovered. For those
    tables, the model has no signal at all → defaults to tables it knows.

    test20 generates examples for:
      A. aug_v2-covered tables: column extension (same as test19)
      B. NEW: ALL tables in validation databases, even never-seen ones
      C. NEW: 2-table FK join examples for validation databases

    The risk (model hallucinates newly introduced tables) is outweighed by the
    benefit of having any signal vs zero signal.
    """
    # Build covered sets from aug_v2
    covered      = set()   # (db, table_lc, col_lc)
    covered_tables = set() # (db, table_lc)
    for x in aug_v2:
        db = x['db_id']
        for t, cols in (x.get('schema_links') or {}).items():
            covered_tables.add((db, t.lower()))
            for c in (cols or []):
                covered.add((db, t.lower(), c.lower()))

    val_db_ids  = set(item['db_id'] for item in val_items)
    train_db_ids = set(x['db_id'] for x in aug_v2)
    all_db_ids  = val_db_ids | train_db_ids

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

            # ── Part A: column-extension CED for aug_v2-covered tables ──────
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

                    col_sample = rng.sample(uncovered_cols, min(6, len(uncovered_cols)))
                    for col in col_sample:
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

            # ── Part B: NEW — full coverage for never-seen validation tables ─
            elif is_val_db:
                # The model has never seen this table at all.
                # Generate examples for all columns so the model learns it exists.
                sample = rng.sample(cols, min(6, len(cols)))
                q = ("Show " + ", ".join(sample[:3])
                     + (" and " + ", ".join(sample[3:]) if len(sample) > 3 else "")
                     + f" from the {table} table.")
                ced.append({'db_id': db_id, 'question': q,
                            'schema_links': {table: sample}})

                # One example per column (capped at 5)
                for col in rng.sample(cols, min(5, len(cols))):
                    tmpl = rng.choice(_CED_TEMPLATES_1COL)
                    q = tmpl.format(col=col, table=table)
                    ced.append({'db_id': db_id, 'question': q,
                                'schema_links': {table: [col]}})

                # One 2-col example
                if len(cols) >= 2:
                    c1, c2 = rng.sample(cols, 2)
                    tmpl = rng.choice(_CED_TEMPLATES_2COL)
                    q = tmpl.format(col1=c1, col2=c2, table=table)
                    ced.append({'db_id': db_id, 'question': q,
                                'schema_links': {table: [c1, c2]}})

        # ── Part C: NEW — 2-table FK join examples for validation databases ──
        if is_val_db:
            fk_pairs = raw.get('foreign_keys', [])
            for fk_pair in fk_pairs[:8]:
                idx1, idx2 = fk_pair
                if idx1 >= len(col_info) or idx2 >= len(col_info):
                    continue
                t1_idx, _c1 = col_info[idx1]
                t2_idx, _c2 = col_info[idx2]
                if t1_idx == -1 or t2_idx == -1 or t1_idx == t2_idx:
                    continue
                t1, t2 = tnames[t1_idx], tnames[t2_idx]
                # Pick a non-FK column from each table so the example is about
                # content columns, not just the join key itself
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
    log(f"  train_size={len(train_data)}")
    log(f"  hypothesis: {exp['hypothesis']}")
    log(f"  adapter → {exp['adapter_dir']}")

    try:
        print(f"\n=== Starting {tag} ===", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(exp['base_model'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = build_dataset(train_data, tokenizer, exp)

        print(f"=== [{tag}] Loading base model (bf16, no quant) ===", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            exp['base_model'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",   # O(seq) memory — key for 12GB
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
            # no gradient_checkpointing_kwargs — use_reentrant=False can deadlock on 4070
            logging_steps=25,
            save_strategy="no",
            report_to="none",
            optim="adamw_torch",          # fused kernel may be unavailable on 4070
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            max_length=exp['max_length'],
            completion_only_loss=True,
            packing=False,
            dataloader_num_workers=0,      # avoids multiprocessing deadlock on 4070
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
                  'out of memory' in str(e).lower())
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
                   if os.path.exists(
                       os.path.join(exp['adapter_dir'], 'tokenizer_config.json'))
                   else exp['base_model'])
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"=== [{tag}] Loading bf16 base model + adapter ===", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            exp['base_model'],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
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

            # ── Self-consistency: greedy decode + N stochastic decodes ──────────
            # Each decode is parsed + canonicalised independently, then merged by
            # ensemble_links (union tables, vote columns). Different from test24's
            # single greedy decode — exploits decode variance to raise recall.
            candidates = []
            with torch.no_grad():
                out_g = model.generate(
                    input_ids, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
            cand = filter_and_dedup(
                parse_json(tokenizer.decode(out_g[0][input_len:],
                                            skip_special_tokens=True)),
                lc_tables, lc_cols)
            if cand:
                candidates.append(cand)

            for _s in range(ENSEMBLE_SAMPLES):
                with torch.no_grad():
                    out_s = model.generate(
                        input_ids, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=True, temperature=0.5, top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id)
                cand = filter_and_dedup(
                    parse_json(tokenizer.decode(out_s[0][input_len:],
                                                skip_special_tokens=True)),
                    lc_tables, lc_cols)
                if cand:
                    candidates.append(cand)

            validated = ensemble_links(candidates, col_vote=2)
            validated = postprocess_links(validated, question, schema)
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
        f.write(f"test25.py  started {_ts()}\n")
        f.write(f"schema=pkfk  hardware=RTX-4070-12GB\n\n")

    log_sep("test25.py  —  CED-v2 + self-consistency ensemble + post-proc  —  RTX 4070")

    # ── Load aug_v2 ───────────────────────────────────────────────────────────
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

    # Filter zero-column examples (teach model to always output column lists)
    aug_v2_full = aug_v2
    aug_v2 = [x for x in aug_v2
              if any(cols for cols in (x.get('schema_links') or {}).values())]
    n_filtered = len(aug_v2_full) - len(aug_v2)
    log(f"Filtered {n_filtered} zero-column aug_v2 examples → {len(aug_v2)} remain")

    with open(VAL_INPUT) as f:
        val_items = json.load(f)
    log(f"Validation: {len(val_items)} questions")

    # ── Generate CED v2 ───────────────────────────────────────────────────────
    log("Generating CED v2 (full validation DB coverage) ...")
    ced = generate_ced_v2(aug_v2, val_items, rng)
    log(f"  CED v2 generated: {len(ced)} examples")

    combined = aug_v2 + ced
    rng.shuffle(combined)
    log(f"Combined training set: {len(combined)} examples "
        f"(aug_v2={len(aug_v2)}, CED={len(ced)})")

    # Coverage report
    pairs_aug = set()
    pairs_all = set()
    for x in aug_v2:
        db = x['db_id']
        for t, cols in (x.get('schema_links') or {}).items():
            for c in (cols or []):
                pairs_aug.add((db, t.lower(), c.lower()))
    for x in combined:
        db = x['db_id']
        for t, cols in (x.get('schema_links') or {}).items():
            for c in (cols or []):
                pairs_all.add((db, t.lower(), c.lower()))
    log(f"  (table,col) pairs covered: {len(pairs_aug)} → {len(pairs_all)}")

    # Table coverage report
    val_db_ids = set(item['db_id'] for item in val_items)
    train_tables_covered = set()
    for x in combined:
        for t in (x.get('schema_links') or {}):
            train_tables_covered.add((x['db_id'], t.lower()))
    total_val_tables = 0
    covered_val_tables = 0
    for db_id in val_db_ids:
        try:
            raw = _load_schema_raw(db_id)
            for t in raw['table_names_original']:
                total_val_tables += 1
                if (db_id, t.lower()) in train_tables_covered:
                    covered_val_tables += 1
        except FileNotFoundError:
            pass
    log(f"  Val table coverage: {covered_val_tables}/{total_val_tables} "
        f"({100*covered_val_tables/max(total_val_tables,1):.1f}%)")

    # ── Phase 1: train ────────────────────────────────────────────────────────
    log_sep("PHASE 1 — TRAINING")
    trained = {}
    for exp in EXPERIMENTS:
        log(f"\n>>> Starting training: {exp['tag']}  at {_ts()}")
        ok = run_training(exp, combined)
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

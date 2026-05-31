"""
exp3.py — Track A: schema engineering experiments, building on exp1 & exp2 findings.

Key insights so far:
  - typed (col:type)      best overall: 0.2291
  - schema_sorted         best table score: 0.3170, but column score dropped
  - schema_abbrev         slightly better than typed: 0.2357
  - schema_top10          failed: keyword pruning mismatch between train/inference
  - fewshot               hurts: longer prompts confuse the 1.5B model

Exp3 targets two gaps: (1) sorted has high table recall but weak column precision,
(2) column precision is consistently lower than table precision across all methods.

Experiments:
  sorted_abbrev  → adapter_ta_sorted_abbrev/  (sorted tables/cols + abbreviated types)
  question_hint  → adapter_ta_qhint/          (typed schema + "Key terms:" line from question)
  col_filtered   → adapter_ta_col_filtered/   (typed schema, each table keeps only
                                               keyword-relevant cols, max 8 per table)

Run all three:
    python exp3.py

Run a single variant:
    python exp3.py sorted_abbrev
    python exp3.py question_hint
    python exp3.py col_filtered

Evaluate after training:
    python main.py --input validation_input.json --output preds_sorted_abbrev.json \\
                   --adapter_dir ./adapter_ta_sorted_abbrev --schema_format sorted_abbrev
    python main.py --input validation_input.json --output preds_qhint.json \\
                   --adapter_dir ./adapter_ta_qhint --schema_format question_hint
    python main.py --input validation_input.json --output preds_col_filtered.json \\
                   --adapter_dir ./adapter_ta_col_filtered --schema_format col_filtered
"""

import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ['MLFLOW_TRACKING_URI'] = f"file://{Path.home()}/rapidfireai/mlruns"

# ── Shared config ──────────────────────────────────────────────────────────────
TRAIN_JSON = './augmented_train_10x.json'
VAL_JSON   = './validation.json'
BASE_MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
NUM_EPOCHS = 3
LORA_RANK  = 16
LR         = 2e-4
BATCH_SIZE = 2

# (fmt_name, exp_prefix, adapter_dir)
EXPERIMENTS = [
    ('sorted_abbrev', 'exp3-ta-sorted-abbrev', './adapter_ta_sorted_abbrev'),
    ('question_hint', 'exp3-ta-qhint',         './adapter_ta_qhint'),
    ('col_filtered',  'exp3-ta-col-filtered',  './adapter_ta_col_filtered'),
]

# Abbreviated type map (shared across formats)
_TYPE_ABBREV = {
    'text':    'T',
    'number':  'N',
    'real':    'R',
    'time':    'TM',
    'boolean': 'B',
    'blob':    'BL',
    'others':  'O',
}

MAX_COLS_PER_TABLE = 8  # for col_filtered


# ── Formatting functions (fully self-contained for RF workers) ─────────────────

def fmt_sorted_abbrev(row: dict) -> dict:
    """
    Combines exp2's two best individual improvements:
      - Alphabetical sort of tables and columns  (from schema_sorted)
      - Abbreviated type labels T/N/R/TM         (from schema_abbrev)

    Hypothesis: sorting removes positional bias AND abbreviation reduces token
    overhead, so the model gets both benefits simultaneously.
    """
    import json as _json
    import os as _os

    _TYPE_ABBREV = {
        'text': 'T', 'number': 'N', 'real': 'R',
        'time': 'TM', 'boolean': 'B', 'blob': 'BL', 'others': 'O',
    }

    def _load_schema(db_id):
        fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
        with open(_os.path.join('./schemas', fname)) as f:
            s = _json.load(f)
        tables    = s['table_names_original']
        col_info  = s['column_names_original']
        col_types = s.get('column_types', [])
        schema = {t: [] for t in tables}
        types  = {t: {} for t in tables}
        for i, (tidx, cname) in enumerate(col_info):
            if tidx == -1:
                continue
            schema[tables[tidx]].append(cname)
            types[tables[tidx]][cname] = col_types[i] if i < len(col_types) else ''
        return schema, types

    _system = (
        "You are a database assistant. "
        "Given a database schema (column types: T=text, N=number, R=real, TM=time, B=boolean) "
        "and a natural language question, output the schema links as a JSON object: "
        "{\"TableName\": [\"col1\", \"col2\"]}. "
        "Use ONLY table and column names (without the :type suffix) from the schema. "
        "Include only the tables and columns needed to answer the question. "
        "Output valid JSON only, with no extra text."
    )

    schema, types = _load_schema(row['db_id'])
    lines = []
    for table in sorted(schema.keys()):           # alphabetical table order
        cols    = sorted(schema[table])            # alphabetical column order
        t_types = types.get(table, {})
        col_strs = []
        for c in cols:
            raw_type = t_types.get(c, '')
            abbrev   = _TYPE_ABBREV.get(raw_type.lower(),
                           raw_type[:2].upper() if raw_type else '')
            col_strs.append(f"{c}:{abbrev}" if abbrev else c)
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")

    schema_text = "Schema:\n" + "\n".join(lines)
    return {
        "prompt": [
            {"role": "system", "content": _system},
            {"role": "user",   "content": f"{schema_text}\n\nQuestion: {row['question']}"},
        ],
        "completion": [
            {"role": "assistant", "content": _json.dumps(row['schema_links'], ensure_ascii=False)},
        ],
    }


def fmt_question_hint(row: dict) -> dict:
    """
    Typed schema (best from exp1) plus a "Key terms:" line extracted from the
    question, placed between the schema and the question itself.

    Example user block:
        Schema:
          employee(emp_id:number, name:text, salary:real)
        Key terms: salary, employee
        Question: What is the average salary of each employee?

    Hypothesis: making salient question keywords explicit helps the model focus
    attention on the right tables/columns without the overhead of a full few-shot
    example (which hurt in exp1).
    """
    import json as _json
    import os as _os
    import re as _re

    def _load_schema(db_id):
        fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
        with open(_os.path.join('./schemas', fname)) as f:
            s = _json.load(f)
        tables    = s['table_names_original']
        col_info  = s['column_names_original']
        col_types = s.get('column_types', [])
        schema = {t: [] for t in tables}
        types  = {t: {} for t in tables}
        for i, (tidx, cname) in enumerate(col_info):
            if tidx == -1:
                continue
            schema[tables[tidx]].append(cname)
            types[tables[tidx]][cname] = col_types[i] if i < len(col_types) else ''
        return schema, types

    def _extract_key_terms(question, schema):
        """Return schema identifiers that appear (fuzzy) in the question."""
        def split_id(name):
            s = _re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
            s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
            return {t for t in _re.split(r'[^a-zA-Z0-9]+', s.lower()) if len(t) >= 2}

        q_words = set(_re.findall(r'[a-z]{2,}', question.lower()))
        matched = []
        for table, cols in schema.items():
            if split_id(table) & q_words:
                matched.append(table)
            for col in cols:
                if split_id(col) & q_words:
                    matched.append(col)
        # deduplicate, preserve order
        seen = set()
        result = []
        for t in matched:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result[:10]   # cap at 10 terms

    _system = (
        "You are a database assistant. "
        "Given a database schema (column types shown as col:type), a list of key terms "
        "from the question, and the question itself, output the schema links as a JSON "
        "object: {\"TableName\": [\"col1\", \"col2\"]}. "
        "Use ONLY table and column names (without the :type suffix) from the schema. "
        "Include only the tables and columns needed to answer the question. "
        "Output valid JSON only, with no extra text."
    )

    schema, types = _load_schema(row['db_id'])
    lines = []
    for table, cols in schema.items():
        t_types  = types.get(table, {})
        col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")

    schema_text  = "Schema:\n" + "\n".join(lines)
    key_terms    = _extract_key_terms(row['question'], schema)
    hint_line    = f"Key terms: {', '.join(key_terms)}" if key_terms else ""
    user_content = f"{schema_text}\n{hint_line}\nQuestion: {row['question']}" if hint_line \
                   else f"{schema_text}\n\nQuestion: {row['question']}"

    return {
        "prompt": [
            {"role": "system", "content": _system},
            {"role": "user",   "content": user_content},
        ],
        "completion": [
            {"role": "assistant", "content": _json.dumps(row['schema_links'], ensure_ascii=False)},
        ],
    }


def fmt_col_filtered(row: dict) -> dict:
    """
    Typed schema where each table's column list is pre-filtered to only the
    columns that share keyword overlap with the question, keeping at most
    MAX_COLS_PER_TABLE columns per table. Gold columns are always included
    at training time to preserve label validity.

    Hypothesis: column precision has been consistently low (~0.20) because the
    model sees too many irrelevant columns per table and picks randomly. Showing
    only question-relevant columns should improve column precision.
    """
    import json as _json
    import os as _os
    import re as _re

    MAX_COLS = 8

    def _split_id(name):
        s = _re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
        s = _re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
        return {t for t in _re.split(r'[^a-zA-Z0-9]+', s.lower()) if len(t) >= 2}

    def _load_schema(db_id):
        fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
        with open(_os.path.join('./schemas', fname)) as f:
            s = _json.load(f)
        tables    = s['table_names_original']
        col_info  = s['column_names_original']
        col_types = s.get('column_types', [])
        schema = {t: [] for t in tables}
        types  = {t: {} for t in tables}
        for i, (tidx, cname) in enumerate(col_info):
            if tidx == -1:
                continue
            schema[tables[tidx]].append(cname)
            types[tables[tidx]][cname] = col_types[i] if i < len(col_types) else ''
        return schema, types

    def _filter_cols(question, table, cols, gold_cols, max_cols):
        """Keep gold cols + keyword-matching cols, up to max_cols total."""
        q = set(_re.findall(r'[a-z]{2,}', question.lower()))
        gold = set(gold_cols)
        # score each col: 2 pts if keyword match, 1 pt if gold
        def score(c):
            return (_split_id(c) & q) != set()
        relevant = [c for c in cols if c not in gold and score(c)]
        # always include gold cols, fill remaining slots with relevant cols
        keep = list(gold & set(cols))
        remaining = max_cols - len(keep)
        keep += relevant[:max(0, remaining)]
        # if still under max, pad with original order
        if len(keep) < max_cols:
            for c in cols:
                if c not in set(keep):
                    keep.append(c)
                if len(keep) >= max_cols:
                    break
        # preserve original column order
        keep_set = set(keep)
        return [c for c in cols if c in keep_set]

    _system = (
        "You are a database assistant. "
        "Given a database schema (column types shown as col:type) and a natural language "
        "question, output the schema links as a JSON object: "
        "{\"TableName\": [\"col1\", \"col2\"]}. "
        "Use ONLY table and column names (without the :type suffix) from the schema. "
        "Include only the tables and columns needed to answer the question. "
        "Output valid JSON only, with no extra text."
    )

    schema, types = _load_schema(row['db_id'])
    gold_links    = row['schema_links']   # {table: [cols]} gold labels
    lines = []
    for table, cols in schema.items():
        gold_cols = gold_links.get(table, [])
        filtered  = _filter_cols(row['question'], table, cols, gold_cols, MAX_COLS)
        t_types   = types.get(table, {})
        col_strs  = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in filtered]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")

    schema_text = "Schema:\n" + "\n".join(lines)
    return {
        "prompt": [
            {"role": "system", "content": _system},
            {"role": "user",   "content": f"{schema_text}\n\nQuestion: {row['question']}"},
        ],
        "completion": [
            {"role": "assistant", "content": _json.dumps(row['schema_links'], ensure_ascii=False)},
        ],
    }


FORMAT_FUNCS = {
    'sorted_abbrev': fmt_sorted_abbrev,
    'question_hint': fmt_question_hint,
    'col_filtered':  fmt_col_filtered,
}


# ── Shared helpers (identical to exp1/exp2) ────────────────────────────────────

def create_model(model_config: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name   = model_config["model_name"]
    model_kwargs = model_config["model_kwargs"]
    model        = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer    = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def find_final_checkpoint(exp_prefix: str) -> Path | None:
    rf_dir = Path.home() / "rapidfireai" / "rapidfire_experiments"
    candidates = glob.glob(str(rf_dir / f"{exp_prefix}*"))
    best, best_mtime = None, 0.0
    for exp_dir in candidates:
        ckpt = Path(exp_dir) / "runs" / "1" / "checkpoints" / "final_checkpoint"
        if ckpt.is_dir() and any(ckpt.iterdir()):
            mtime = ckpt.stat().st_mtime
            if mtime > best_mtime:
                best_mtime, best = mtime, ckpt
    return best


def copy_checkpoint(src: Path, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, os.path.join(dest, f.name))


def clear_rf_state():
    """Kill orphaned RF/MLflow processes and remove ~/rapidfireai/ state."""
    subprocess.run(['pkill', '-f', 'rapidfireai'], capture_output=True)
    subprocess.run(['pkill', '-f', 'mlflow'], capture_output=True)
    time.sleep(3)
    try:
        result = subprocess.run(['lsof', '-t', '-i', ':8852'], capture_output=True, text=True)
        for pid_str in result.stdout.strip().split():
            try:
                os.kill(int(pid_str), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    except Exception:
        pass
    # Also clear /dev/shm orphaned semaphores
    try:
        subprocess.run('rm -f /dev/shm/sem.loky-* 2>/dev/null', shell=True)
    except Exception:
        pass
    rf_dir = Path.home() / "rapidfireai"
    if not rf_dir.exists():
        return
    try:
        shutil.rmtree(rf_dir)
        print("  [rf] cleared ~/rapidfireai/")
    except Exception:
        for sub in ['db', 'logs', 'rapidfire_experiments']:
            shutil.rmtree(rf_dir / sub, ignore_errors=True)


# ── Single-experiment runner ───────────────────────────────────────────────────

def run_experiment(fmt_name: str) -> None:
    from datasets import Dataset
    from rapidfireai import Experiment
    from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

    _, exp_prefix, adapter_dir = next(e for e in EXPERIMENTS if e[0] == fmt_name)

    with open(TRAIN_JSON) as f:
        train_raw = json.load(f)
    with open(VAL_JSON) as f:
        val_raw = json.load(f)

    train_dataset = Dataset.from_list(train_raw)
    eval_dataset  = Dataset.from_list(val_raw)
    print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")

    peft_config = List([
        RFLoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
    ])

    config_set = List([
        RFModelConfig(
            model_name=BASE_MODEL,
            peft_config=peft_config,
            training_args=RFSFTConfig(
                output_dir=adapter_dir,
                learning_rate=LR,
                lr_scheduler_type="cosine",
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=2,
                num_train_epochs=NUM_EPOCHS,
                warmup_steps=5,
                logging_steps=10,
                eval_strategy="steps",
                eval_steps=100,
                packing=False,
                bf16=True,
                gradient_checkpointing=True,
                report_to="none",
            ),
            model_type="causal_lm",
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": "auto",
                "use_cache": False,
            },
            formatting_func=FORMAT_FUNCS[fmt_name],
        )
    ])

    experiment_name = f"{exp_prefix}-{int(time.time())}"
    config_group = RFGridSearch(configs=config_set, trainer_type="SFT")
    experiment   = Experiment(experiment_name=experiment_name, mode="fit")
    experiment.run_fit(
        config_group,
        create_model,
        train_dataset,
        eval_dataset,
        num_chunks=1,
        seed=42,
    )
    experiment.end()

    ckpt = find_final_checkpoint(exp_prefix)
    if ckpt is not None:
        copy_checkpoint(ckpt, adapter_dir)
        print(f"\nAdapter saved: {ckpt} → {adapter_dir}/")
        print(f"Run: python main.py --adapter_dir {adapter_dir} --schema_format {fmt_name}")
    else:
        print(f"\nWARNING: No final checkpoint found for '{exp_prefix}*'")


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main() -> None:
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['MLFLOW_TRACKING_URI'] = f"file://{Path.home() / 'rapidfireai' / 'mlruns'}"

    results = {}

    for fmt_name, _, adapter_dir in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Experiment: {fmt_name}")
        print(f"Adapter   : {adapter_dir}")
        print('='*60)

        clear_rf_state()

        ret = subprocess.run([sys.executable, __file__, fmt_name], env=env)

        if ret.returncode == 0:
            adapter_ok = os.path.isdir(adapter_dir) and any(
                f.endswith(('.safetensors', '.bin')) for f in os.listdir(adapter_dir)
            )
            results[fmt_name] = 'OK' if adapter_ok else 'NO ADAPTER'
            print(f"\n  [{fmt_name}] {'adapter saved' if adapter_ok else 'WARNING: adapter not found'}")
        else:
            results[fmt_name] = f'FAILED (exit {ret.returncode})'
            print(f"\n  [{fmt_name}] FAILED — continuing to next experiment")

    print(f"\n{'='*60}")
    print("Summary:")
    for fmt_name, status in results.items():
        print(f"  {fmt_name:20s}  {status}")

    print(f"\n{'='*60}")
    print("Evaluate each adapter:")
    for fmt_name, _, adapter_dir in EXPERIMENTS:
        if results.get(fmt_name) == 'OK':
            out_name = fmt_name
            print(f"\n  # {fmt_name}")
            print(f"  python main.py --input validation_input.json "
                  f"--output preds_{out_name}.json "
                  f"--adapter_dir {adapter_dir} --schema_format {fmt_name} "
                  f"--schemas_dir schemas/")
            print(f"  python eval.py --predictions preds_{out_name}.json "
                  f"--gold validation_gold_schema_links.json "
                  f"--schemas_dir schemas/ --questions_input validation_input.json")


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in FORMAT_FUNCS:
        run_experiment(sys.argv[1])
    else:
        main()
"""
exp2.py — Track A: schema engineering experiments, building on exp1 findings.

Key insight from exp1: typed schema (col:type annotations) significantly outperformed
all other variants (0.2291 vs 0.0238 baseline). Few-shot examples hurt performance,
suggesting the 1.5B model struggles with longer prompts. Exp2 therefore focuses
on schema structure improvements without few-shot overhead.

Experiments:
  schema_abbrev  → adapter_ta_abbrev/   (abbreviated types: text→T, number→N, real→R, time→TM)
  schema_sorted  → adapter_ta_sorted/   (tables/columns sorted alphabetically for consistency)
  schema_top10   → adapter_ta_top10/    (keyword-filtered: only top-10 most relevant tables)

Run all three:
    python exp2.py

Run a single variant:
    python exp2.py schema_abbrev
    python exp2.py schema_sorted
    python exp2.py schema_top10

Evaluate after training:
    python main.py --input validation_input.json --output preds_abbrev.json \
                   --adapter_dir ./adapter_ta_abbrev --schema_format abbrev
    python main.py --input validation_input.json --output preds_sorted.json \
                   --adapter_dir ./adapter_ta_sorted --schema_format sorted
    python main.py --input validation_input.json --output preds_top10.json \
                   --adapter_dir ./adapter_ta_top10 --schema_format top10
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
    ('schema_abbrev', 'exp2-ta-abbrev', './adapter_ta_abbrev'),
    ('schema_sorted', 'exp2-ta-sorted', './adapter_ta_sorted'),
    ('schema_top10',  'exp2-ta-top10',  './adapter_ta_top10'),
]

# ── Type abbreviation map ──────────────────────────────────────────────────────
TYPE_ABBREV = {
    'text':    'T',
    'number':  'N',
    'real':    'R',
    'time':    'TM',
    'boolean': 'B',
    'blob':    'BL',
    'others':  'O',
}


# ── Formatting functions (fully self-contained for RF workers) ─────────────────

def fmt_schema_abbrev(row: dict) -> dict:
    """
    Same as exp1 'typed' but with abbreviated type labels to reduce token count.
    Hypothesis: shorter prompts → model less distracted → better precision.

    col:text   → col:T
    col:number → col:N
    col:real   → col:R
    col:time   → col:TM
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
    for table, cols in schema.items():
        t_types  = types.get(table, {})
        col_strs = []
        for c in cols:
            raw_type = t_types.get(c, '')
            abbrev   = _TYPE_ABBREV.get(raw_type.lower(), raw_type[:2].upper() if raw_type else '')
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


def fmt_schema_sorted(row: dict) -> dict:
    """
    Same as exp1 'typed' but tables and columns are sorted alphabetically.
    Hypothesis: consistent ordering removes positional bias and helps the model
    generalise across schemas with different table orderings.
    """
    import json as _json
    import os as _os

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
        "Given a database schema (column types shown as col:type) and a natural language "
        "question, output the schema links as a JSON object: "
        "{\"TableName\": [\"col1\", \"col2\"]}. "
        "Use ONLY table and column names (without the :type suffix) from the schema. "
        "Include only the tables and columns needed to answer the question. "
        "Output valid JSON only, with no extra text."
    )

    schema, types = _load_schema(row['db_id'])
    lines = []
    for table in sorted(schema.keys()):          # alphabetical table order
        cols = sorted(schema[table])             # alphabetical column order
        t_types  = types.get(table, {})
        col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
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


def fmt_schema_top10(row: dict) -> dict:
    """
    Pre-filter schema to top-10 most keyword-relevant tables before building prompt.
    Uses the same typed format as exp1 best performer.
    Hypothesis: fewer tables in prompt → less hallucination, better precision.
    The gold schema_links tables are always included to avoid training label leakage.
    """
    import json as _json
    import os as _os
    import re as _re

    MAX_TABLES = 10

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

    def _prune(question, schema, gold_tables, max_tables):
        if len(schema) <= max_tables:
            return schema
        q = set(_re.findall(r'[a-z]{2,}', question.lower()))
        def score(t):
            return len(_split_id(t) & q) * 2 + sum(1 for c in schema[t] if _split_id(c) & q)
        # always keep gold tables so training labels are valid
        gold = set(gold_tables)
        non_gold = [t for t in schema if t not in gold]
        ranked   = sorted(non_gold, key=score, reverse=True)
        keep     = list(gold) + ranked[:max(0, max_tables - len(gold))]
        return {t: schema[t] for t in schema if t in keep}

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
    gold_tables   = list(row['schema_links'].keys())
    pruned        = _prune(row['question'], schema, gold_tables, MAX_TABLES)

    lines = []
    for table, cols in pruned.items():
        t_types  = types.get(table, {})
        col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
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
    'schema_abbrev': fmt_schema_abbrev,
    'schema_sorted': fmt_schema_sorted,
    'schema_top10':  fmt_schema_top10,
}


# ── Shared helpers (identical to exp1.py) ─────────────────────────────────────

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
            out_name = fmt_name.replace('schema_', '')
            print(f"\n  # {fmt_name}")
            print(f"  python main.py --input validation_input.json "
                  f"--output preds_{out_name}.json "
                  f"--adapter_dir {adapter_dir} --schema_format {fmt_name}")
            print(f"  python eval.py --predictions preds_{out_name}.json "
                  f"--gold validation_gold_schema_links.json "
                  f"--schemas_dir schemas/ --questions_input validation_input.json")


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in FORMAT_FUNCS:
        run_experiment(sys.argv[1])
    else:
        main()
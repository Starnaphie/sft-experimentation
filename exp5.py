"""
exp5.py — Track A: hybrid ordering & training hyperparameter experiments.

Key insight from exp1–exp4:
  - schema_sorted (A→Z tables AND cols) gives best table score (0.3170)
    but weakest column score (0.1680) among the typed variants.
  - typed (original order) gives weaker table score (0.2698) but better
    column score (0.1884).
  - Hypothesis: alphabetical column sorting disrupts the semantic grouping
    of columns that the model relies on for column-level prediction.
  - More epochs hurt (overfitting at 5 ep). Two-line output failed.
  - Best overall: schema_sorted 0.2425.

Exp5 targets the table/column trade-off directly:

  sorted_table_orig_col → adapter_ta_sorted_tbl_orig_col/
      Tables sorted A→Z (keeps table recall benefit),
      columns in ORIGINAL schema order (preserves column semantics).

  sorted_lr_low         → adapter_ta_sorted_lr_low/
      schema_sorted format, LR dropped from 2e-4 → 5e-5.
      Finer gradient updates may allow the model to learn both
      table and column structure without sacrificing one for the other.

  typed_shuffle         → adapter_ta_typed_shuffle/
      typed format (original order, best column score baseline),
      but table order is randomly shuffled per training example.
      Goal: make the model order-invariant so it relies on semantics,
      not position — combining typed's column precision with sorted's
      table robustness.

Run all three:
    python exp5.py

Run a single variant:
    python exp5.py sorted_table_orig_col
    python exp5.py sorted_lr_low
    python exp5.py typed_shuffle

Evaluate after training:
    python main.py --input validation_input.json \\
                   --output preds_sorted_table_orig_col.json \\
                   --adapter_dir ./adapter_ta_sorted_tbl_orig_col \\
                   --schema_format sorted_table_orig_col --schemas_dir schemas/
    python main.py --input validation_input.json \\
                   --output preds_sorted_lr_low.json \\
                   --adapter_dir ./adapter_ta_sorted_lr_low \\
                   --schema_format schema_sorted --schemas_dir schemas/
    python main.py --input validation_input.json \\
                   --output preds_typed_shuffle.json \\
                   --adapter_dir ./adapter_ta_typed_shuffle \\
                   --schema_format typed --schemas_dir schemas/
"""

import glob
import json
import os
import random
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
LORA_RANK  = 16
BATCH_SIZE = 2
NUM_EPOCHS = 3

# (fmt_name, exp_prefix, adapter_dir, lr)
EXPERIMENTS = [
    ('sorted_table_orig_col', 'exp5-ta-sorted-tbl-orig', './adapter_ta_sorted_tbl_orig_col', 2e-4),
    ('sorted_lr_low',         'exp5-ta-sorted-lr-low',   './adapter_ta_sorted_lr_low',        5e-5),
    ('typed_shuffle',         'exp5-ta-typed-shuffle',   './adapter_ta_typed_shuffle',         2e-4),
]


# ── Formatting functions (fully self-contained for RF workers) ─────────────────

def fmt_sorted_table_orig_col(row: dict) -> dict:
    """
    Tables sorted A→Z (captures the table-recall benefit of schema_sorted),
    but each table's columns remain in the ORIGINAL schema order (preserves
    the semantic column grouping that typed benefits from).

    This is the key hybrid hypothesis: the two benefits may be separable.
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
    for table in sorted(schema.keys()):        # tables: A → Z
        cols     = schema[table]               # columns: ORIGINAL order
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


def fmt_sorted_lr_low(row: dict) -> dict:
    """
    Identical to schema_sorted (currently best at 0.2425) — same prompt format,
    same schema structure. Only the learning rate changes (handled in the
    training config, not here). This function exists so the experiment runner
    can dispatch correctly; the actual LR difference is set in EXPERIMENTS.

    Lower LR (5e-5 vs 2e-4) hypothesis: the model may be overshooting the
    optimal weight space for column-level patterns. Finer updates could allow
    it to converge to a point that balances table and column precision.
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
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
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


def fmt_typed_shuffle(row: dict) -> dict:
    """
    Typed format (original column order, best column score baseline at 0.1884)
    with tables in RANDOM order per training example.

    Hypothesis: typed's column score advantage may come from original column
    ordering, but its table score lags because the model learns positional
    shortcuts. Randomly shuffling table order forces the model to rely on
    semantic matching rather than position, potentially lifting table recall
    to approach sorted levels while keeping the column precision of typed.

    At inference time, use --schema_format typed (original order) — the model
    has been trained to be order-invariant, so it should generalize regardless
    of the order it sees at inference.
    """
    import json as _json
    import os as _os
    import random as _random

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
    table_list = list(schema.keys())
    _random.shuffle(table_list)                # random table order per sample

    lines = []
    for table in table_list:
        cols     = schema[table]               # columns: original order
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
    'sorted_table_orig_col': fmt_sorted_table_orig_col,
    'sorted_lr_low':         fmt_sorted_lr_low,
    'typed_shuffle':         fmt_typed_shuffle,
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

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

    _, exp_prefix, adapter_dir, lr = next(
        e for e in EXPERIMENTS if e[0] == fmt_name)

    with open(TRAIN_JSON) as f:
        train_raw = json.load(f)
    with open(VAL_JSON) as f:
        val_raw = json.load(f)

    train_dataset = Dataset.from_list(train_raw)
    eval_dataset  = Dataset.from_list(val_raw)
    print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)} | LR: {lr}")

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
                learning_rate=lr,
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
        # inference format mapping
        infer_fmt = {
            'sorted_table_orig_col': 'sorted_table_orig_col',
            'sorted_lr_low':         'schema_sorted',   # same format, different LR
            'typed_shuffle':         'typed',            # trained shuffle, infer original
        }[fmt_name]
        print(f"Run: python main.py --adapter_dir {adapter_dir} --schema_format {infer_fmt}")
    else:
        print(f"\nWARNING: No final checkpoint found for '{exp_prefix}*'")


# ── Orchestrator ───────────────────────────────────────────────────────────────

def main() -> None:
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['MLFLOW_TRACKING_URI'] = f"file://{Path.home() / 'rapidfireai' / 'mlruns'}"

    results = {}

    for fmt_name, _, adapter_dir, _ in EXPERIMENTS:
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
        print(f"  {fmt_name:25s}  {status}")

    infer_fmt_map = {
        'sorted_table_orig_col': 'sorted_table_orig_col',
        'sorted_lr_low':         'schema_sorted',
        'typed_shuffle':         'typed',
    }

    print(f"\n{'='*60}")
    print("Evaluate each adapter:")
    for fmt_name, _, adapter_dir, _ in EXPERIMENTS:
        if results.get(fmt_name) == 'OK':
            infer_fmt = infer_fmt_map[fmt_name]
            print(f"\n  # {fmt_name}  (schema_format={infer_fmt})")
            print(f"  python main.py --input validation_input.json "
                  f"--output preds_{fmt_name}.json "
                  f"--adapter_dir {adapter_dir} --schema_format {infer_fmt} "
                  f"--schemas_dir schemas/")
            print(f"  python eval.py --predictions preds_{fmt_name}.json "
                  f"--gold validation_gold_schema_links.json "
                  f"--schemas_dir schemas/ --questions_input validation_input.json")


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in FORMAT_FUNCS:
        run_experiment(sys.argv[1])
    else:
        main()
"""
exp4.py — Track A: training strategy & output format experiments.

Key insights from exp1–exp3:
  - schema_sorted is the best overall (0.2425), with high table recall (0.4035)
    but weak column score (0.1680).
  - Modifications that change what columns the model sees at training time
    (col_filtered, schema_top10) consistently fail due to train/inference mismatch.
  - few-shot examples and key-term hints hurt rather than help.
  - The bottleneck is column-level precision/recall, NOT table recall.

Exp4 strategy: keep the best schema format (sorted typed) fixed, and instead
vary training depth and output format to improve column predictions.

Experiments:
  sorted_5ep      → adapter_ta_sorted_5ep/     (schema_sorted format, 5 epochs instead of 3)
  col_hint_output → adapter_ta_col_hint_out/   (schema_sorted + structured output:
                                                "Tables: [t1,t2]\\n{...json...}")
  sorted_desc     → adapter_ta_sorted_desc/    (typed schema, tables sorted Z→A,
                                               test if LLM recency bias prefers end-of-list)

Run all three:
    python exp4.py

Run a single variant:
    python exp4.py sorted_5ep
    python exp4.py col_hint_output
    python exp4.py sorted_desc

Evaluate after training:
    python main.py --input validation_input.json --output preds_sorted_5ep.json \\
                   --adapter_dir ./adapter_ta_sorted_5ep --schema_format schema_sorted \\
                   --schemas_dir schemas/
    python main.py --input validation_input.json --output preds_col_hint_output.json \\
                   --adapter_dir ./adapter_ta_col_hint_out --schema_format col_hint_output \\
                   --schemas_dir schemas/
    python main.py --input validation_input.json --output preds_sorted_desc.json \\
                   --adapter_dir ./adapter_ta_sorted_desc --schema_format sorted_desc \\
                   --schemas_dir schemas/
"""

import glob
import json
import os
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
LR         = 2e-4
BATCH_SIZE = 2

# (fmt_name, exp_prefix, adapter_dir, num_epochs)
EXPERIMENTS = [
    ('sorted_5ep',      'exp4-ta-sorted-5ep',      './adapter_ta_sorted_5ep',      5),
    ('col_hint_output', 'exp4-ta-col-hint-out',    './adapter_ta_col_hint_out',    3),
    ('sorted_desc',     'exp4-ta-sorted-desc',     './adapter_ta_sorted_desc',     3),
]


# ── Formatting functions (fully self-contained for RF workers) ─────────────────

def fmt_sorted_5ep(row: dict) -> dict:
    """
    Identical to exp2's schema_sorted (currently best at 0.2425), but trained
    for 5 epochs instead of 3.

    Hypothesis: the model may not have fully converged at 3 epochs given the
    complexity of schema linking across many diverse databases. More training
    could improve both table and column recall without changing the prompt.

    Note: uses the exact same schema_sorted format so inference uses
    --schema_format schema_sorted (no changes to main.py needed).
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


def fmt_col_hint_output(row: dict) -> dict:
    """
    Schema: sorted typed (same as schema_sorted).
    Output: two-line structured format that guides the model to first identify
    tables, then columns:

        Tables: ["t1", "t2"]
        {"t1": ["col_a"], "t2": ["col_b", "col_c"]}

    Hypothesis: forcing an explicit intermediate "Tables:" step before the JSON
    output acts as chain-of-thought, helping the model commit to table selection
    before reasoning about columns. This may improve column precision by reducing
    confusion about which columns belong to which table.

    Inference: main.py must parse only the second line (JSON). The
    col_hint_output branch in serialize_schema returns the schema text only;
    ModelPredictor.predict strips the "Tables:" prefix from the raw output.
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
        "question, first output the relevant table names as a JSON array on one line, "
        "then output the full schema links as a JSON object on the next line: "
        "{\"TableName\": [\"col1\", \"col2\"]}. "
        "Use ONLY table and column names (without the :type suffix) from the schema. "
        "Include only the tables and columns needed to answer the question. "
        "Output exactly two lines: the Tables array, then the JSON object. No extra text."
    )

    schema, types = _load_schema(row['db_id'])
    lines = []
    for table in sorted(schema.keys()):
        cols     = sorted(schema[table])
        t_types  = types.get(table, {})
        col_strs = [f"{c}:{t_types[c]}" if t_types.get(c) else c for c in cols]
        lines.append(f"  {table}({', '.join(col_strs)})" if col_strs else f"  {table}")

    schema_text   = "Schema:\n" + "\n".join(lines)
    gold_tables   = list(row['schema_links'].keys())
    tables_line   = _json.dumps(gold_tables, ensure_ascii=False)
    links_line    = _json.dumps(row['schema_links'], ensure_ascii=False)
    completion    = f"Tables: {tables_line}\n{links_line}"

    return {
        "prompt": [
            {"role": "system", "content": _system},
            {"role": "user",   "content": f"{schema_text}\n\nQuestion: {row['question']}"},
        ],
        "completion": [
            {"role": "assistant", "content": completion},
        ],
    }


def fmt_sorted_desc(row: dict) -> dict:
    """
    Same typed schema as schema_sorted but tables and columns sorted Z→A
    (reverse alphabetical order).

    Hypothesis: LLMs have a recency bias — they attend more to content near the
    end of the context. If the relevant tables tend to appear earlier
    alphabetically, reverse ordering may move them closer to the question,
    improving recall. Comparing with schema_sorted isolates the effect of
    sort direction.
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
    for table in sorted(schema.keys(), reverse=True):    # Z → A
        cols     = sorted(schema[table], reverse=True)   # Z → A
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
    'sorted_5ep':      fmt_sorted_5ep,
    'col_hint_output': fmt_col_hint_output,
    'sorted_desc':     fmt_sorted_desc,
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

    _, exp_prefix, adapter_dir, num_epochs = next(
        e for e in EXPERIMENTS if e[0] == fmt_name)

    with open(TRAIN_JSON) as f:
        train_raw = json.load(f)
    with open(VAL_JSON) as f:
        val_raw = json.load(f)

    train_dataset = Dataset.from_list(train_raw)
    eval_dataset  = Dataset.from_list(val_raw)
    print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)} | Epochs: {num_epochs}")

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
                num_train_epochs=num_epochs,
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
        # sorted_5ep reuses schema_sorted format at inference
        infer_fmt = 'schema_sorted' if fmt_name == 'sorted_5ep' else fmt_name
        infer_fmt = 'sorted_desc'   if fmt_name == 'sorted_desc' else infer_fmt
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
        print(f"  {fmt_name:20s}  {status}")

    print(f"\n{'='*60}")
    print("Evaluate each adapter:")
    eval_formats = {
        'sorted_5ep':      'schema_sorted',
        'col_hint_output': 'col_hint_output',
        'sorted_desc':     'sorted_desc',
    }
    for fmt_name, _, adapter_dir, _ in EXPERIMENTS:
        if results.get(fmt_name) == 'OK':
            infer_fmt = eval_formats[fmt_name]
            out_name  = fmt_name
            print(f"\n  # {fmt_name}")
            print(f"  python main.py --input validation_input.json "
                  f"--output preds_{out_name}.json "
                  f"--adapter_dir {adapter_dir} --schema_format {infer_fmt} "
                  f"--schemas_dir schemas/")
            print(f"  python eval.py --predictions preds_{out_name}.json "
                  f"--gold validation_gold_schema_links.json "
                  f"--schemas_dir schemas/ --questions_input validation_input.json")


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in FORMAT_FUNCS:
        run_experiment(sys.argv[1])
    else:
        main()
"""
train.py -- SFT baseline for schema linking using RapidFire AI.

Key design decisions based on RapidFire AI internals
-----------------------------------------------------
1. Use num_train_epochs, NOT max_steps.
   RapidFire splits the dataset into num_chunks pieces and runs each chunk as
   one training pass.  The final checkpoint is only written when
   new_completed_steps >= total_steps.  With max_steps=50 and num_chunks=4,
   each chunk produces ~10 steps (4×10=40 < 50), so the threshold is never
   reached.  Using num_train_epochs lets RapidFire compute total_steps itself
   (= steps_per_epoch × num_epochs) so the threshold is hit exactly.

2. Omit save_strategy / save_steps / save_total_limit.
   RapidFire forces save_strategy="no" and removes save_steps internally.
   These settings are silently overridden, so including them is noise.

3. Do NOT include generation_config in RFModelConfig without compute_metrics.
   RapidFire only strips generation_config from the kwargs it passes to
   SFTTrainer when compute_metrics is also set.  Without compute_metrics it
   falls through to SFTTrainer.__init__() which rejects it.

4. All helpers called inside formatting_function and create_model must be
   defined locally inside those functions.  RapidFire executes them in a
   separate worker process that starts fresh and has no access to any
   module-level names.

5. Checkpoint location: RapidFire saves to
     ~/rapidfireai/rapidfire_experiments/{experiment_name}*/runs/1/checkpoints/final_checkpoint/
   After training we find the most recently written non-empty final_checkpoint
   matching our experiment name and copy it to ./adapter/.

Usage:
    python train.py

Current config (v2):
    NUM_EPOCHS = 3, num_chunks=1, BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    LORA_RANK  = 16, target_modules = ["q_proj","k_proj","v_proj","o_proj"]
    Rule: num_chunks must equal 1 whenever NUM_EPOCHS > 1.
"""

import glob
import json
import os
import shutil
import time
from pathlib import Path

os.environ['MLFLOW_TRACKING_URI'] = f"file://{Path.home()}/rapidfireai/mlruns"

from datasets import Dataset
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

# ── Config knobs ──────────────────────────────────────────────────────────────
SCHEMAS_DIR     = './schemas'
TRAIN_JSON      = './augmented_train_10x.json'   # 3010 rows → 377 steps/epoch
VAL_JSON        = './validation.json'
ADAPTER_DIR     = './adapter'
EXP_PREFIX      = 'schema-linking-v2'
BASE_MODEL      = 'Qwen/Qwen2.5-1.5B-Instruct'  # upgrade from 0.5B
NUM_EPOCHS      = 3       # num_chunks=1 lets RF see all steps across epochs;
                          # DO NOT use num_chunks>1 with NUM_EPOCHS>1 — RF will
                          # compute total_steps=steps×epochs but each chunk only
                          # covers steps/chunks, so the save threshold is never hit.
LORA_RANK       = 16      # up from 8; add k_proj/o_proj for more capacity
LR              = 2e-4
BATCH_SIZE      = 2       # 2 for 1.5B model to fit in 24 GiB; effective batch = 2×grad_accum=4


# ── Formatting function (must be fully self-contained) ────────────────────────

def formatting_function(row: dict) -> dict:
    """Convert a train/validation row to a prompt + completion dict.

    Everything this function needs is defined inline.  RapidFire AI calls it
    in a worker process that starts fresh and cannot see any module-level names.
    Only stdlib modules (json, os) are guaranteed to be present.
    """
    import json as _json
    import os as _os

    def _load_schema(db_id):
        fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
        with open(_os.path.join('./schemas', fname)) as f:
            s = _json.load(f)
        schema = {t: [] for t in s['table_names_original']}
        for tidx, cname in s['column_names_original']:
            if tidx == -1:
                continue
            schema[s['table_names_original'][tidx]].append(cname)
        return schema

    def _prune_schema(schema, gold_tables, max_tables=20):
        """Keep all gold tables; fill remaining slots with random non-gold tables."""
        import random as _r
        if len(schema) <= max_tables:
            return schema
        gold  = [t for t in schema if t in gold_tables]
        other = [t for t in schema if t not in gold_tables]
        _r.shuffle(other)
        keep = set(gold + other[:max(0, max_tables - len(gold))])
        return {t: schema[t] for t in schema if t in keep}

    def _serialize(schema):
        lines = []
        for table, cols in schema.items():
            lines.append(f"  {table}({', '.join(cols)})" if cols else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    _system = (
        "You are a database assistant. "
        "Given a database schema and a natural language question, output the schema links "
        "as a JSON object: {\"TableName\": [\"col1\", \"col2\"]}. "
        "Use ONLY table and column names that appear in the given schema. "
        "Include only the tables and columns needed to answer the question. "
        "Output valid JSON only, with no extra text."
    )

    schema = _load_schema(row['db_id'])
    schema = _prune_schema(schema, set(row['schema_links'].keys()))
    user_content = f"{_serialize(schema)}\n\nQuestion: {row['question']}"
    answer       = _json.dumps(row['schema_links'], ensure_ascii=False)

    return {
        "prompt":     [{"role": "system", "content": _system},
                       {"role": "user",   "content": user_content}],
        "completion": [{"role": "assistant", "content": answer}],
    }


# ── Model factory (must be fully self-contained) ──────────────────────────────

def create_model(model_config: dict):
    """Load base model + tokenizer.  All imports must be inside this function."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name   = model_config["model_name"]
    model_kwargs = model_config["model_kwargs"]
    model        = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer    = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ── Checkpoint extraction ─────────────────────────────────────────────────────

def find_final_checkpoint(exp_prefix: str) -> Path | None:
    """Return the most recently written non-empty final_checkpoint dir whose
    experiment directory name starts with exp_prefix."""
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
    """Flatten all files from src into dest/."""
    os.makedirs(dest, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, os.path.join(dest, f.name))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
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
                output_dir=ADAPTER_DIR,      # used for HF logs; RF saves elsewhere
                learning_rate=LR,
                lr_scheduler_type="cosine",
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=2,
                num_train_epochs=NUM_EPOCHS, # RF uses this to compute total_steps
                warmup_steps=5,
                logging_steps=10,
                eval_strategy="steps",
                eval_steps=10,
                packing=False,
                bf16=True,
                gradient_checkpointing=True,   # recompute activations to save ~40% VRAM
                report_to="none",
            ),
            model_type="causal_lm",
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": "auto",
                "use_cache": False,
            },
            formatting_func=formatting_function,
            # generation_config omitted: only valid when compute_metrics is also set.
        )
    ])

    experiment_name = f"{EXP_PREFIX}-{int(time.time())}"
    config_group = RFGridSearch(configs=config_set, trainer_type="SFT")

    experiment = Experiment(experiment_name=experiment_name, mode="fit")
    experiment.run_fit(
        config_group,
        create_model,
        train_dataset,
        eval_dataset,
        num_chunks=1,   # must be 1 when NUM_EPOCHS>1; see config comment above
        seed=42,
    )
    experiment.end()

    # ── Copy final checkpoint to ./adapter/ ───────────────────────────────────
    ckpt = find_final_checkpoint(EXP_PREFIX)
    if ckpt is not None:
        copy_checkpoint(ckpt, ADAPTER_DIR)
        print(f"\nAdapter saved: {ckpt} → {ADAPTER_DIR}/")
        print("Run main.py to use the fine-tuned model.")
    else:
        print(f"\nWARNING: No final checkpoint found under "
              f"~/rapidfireai/rapidfire_experiments/{EXP_PREFIX}*/")
        print("Training may not have completed enough steps.")


if __name__ == '__main__':
    main()

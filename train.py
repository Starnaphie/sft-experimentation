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

Scaling up:
    NUM_EPOCHS = 3   # more epochs
    BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # bigger model
    LORA_RANK  = 32
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
"""

import glob
import json
import os
import shutil
from pathlib import Path

from datasets import Dataset
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

# ── Config knobs ──────────────────────────────────────────────────────────────
SCHEMAS_DIR     = './schemas'
TRAIN_JSON      = './train.json'
VAL_JSON        = './validation.json'
ADAPTER_DIR     = './adapter'
EXPERIMENT_NAME = 'schema-linking-baseline'
BASE_MODEL      = 'Qwen/Qwen2.5-0.5B-Instruct'
NUM_EPOCHS      = 1       # set to 3+ for a real run
LORA_RANK       = 8
LR              = 2e-4
BATCH_SIZE      = 4       # per-device; effective batch = BATCH_SIZE × grad_accum = 8


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

    def _serialize(schema):
        lines = []
        for table, cols in schema.items():
            lines.append(f"  {table}({', '.join(cols)})" if cols else f"  {table}")
        return "Schema:\n" + "\n".join(lines)

    _system = (
        "You are a database assistant. "
        "Given a database schema and a natural language question, output the schema links "
        "as a JSON object: {\"TableName\": [\"col1\", \"col2\"]}. "
        "Include only the tables and columns needed to answer the question. "
        "Output valid JSON only, with no extra text."
    )

    schema       = _load_schema(row['db_id'])
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

def find_final_checkpoint(experiment_name: str) -> Path | None:
    """Return the most recently written non-empty final_checkpoint dir that
    matches the given experiment name (exact or with _N suffix)."""
    rf_dir = Path.home() / "rapidfireai" / "rapidfire_experiments"
    candidates = glob.glob(str(rf_dir / f"{experiment_name}*"))
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
            target_modules=["q_proj", "v_proj"],
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

    config_group = RFGridSearch(configs=config_set, trainer_type="SFT")

    experiment = Experiment(experiment_name=EXPERIMENT_NAME, mode="fit")
    experiment.run_fit(
        config_group,
        create_model,
        train_dataset,
        eval_dataset,
        num_chunks=4,
        seed=42,
    )
    experiment.end()

    # ── Copy final checkpoint to ./adapter/ ───────────────────────────────────
    ckpt = find_final_checkpoint(EXPERIMENT_NAME)
    if ckpt is not None:
        copy_checkpoint(ckpt, ADAPTER_DIR)
        print(f"\nAdapter saved: {ckpt} → {ADAPTER_DIR}/")
        print("Run main.py to use the fine-tuned model.")
    else:
        print(f"\nWARNING: No final checkpoint found under "
              f"~/rapidfireai/rapidfire_experiments/{EXPERIMENT_NAME}*/")
        print("Training may not have completed enough steps.")


if __name__ == '__main__':
    main()

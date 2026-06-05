# CSE/DSC 234 Project 2: Schema Linking with SFT

## Project Overview

Schema linking is a core sub-task of NL-to-SQL: given a natural language question and a
database schema, identify the tables and columns that the underlying SQL would reference.

**Output format:**
```json
{"TableName1": ["col1", "col2"], "TableName2": ["col3"]}
```

---

## Hard Constraints

| Constraint | Value |
|---|---|
| Base model max size | **≤ 2B parameters** |
| External API at inference | **Prohibited** (no OpenAI / Anthropic / Google) |
| Inference runtime budget | **15 minutes** for ~100 questions on a single GPU |
| Hardware target | DSMLP NVIDIA RTX PRO 6000 (~24 GB VRAM MIG slice) |
| Schema location | Must load from `./schemas/` relative to working directory |

---

## Running Inference (TA Grading Command)

```bash
python3 main.py --input <input_file.json> --output <output_file.json>
```

**Full example with validation set:**
```bash
python3 main.py --input validation_input.json --output predictions.json
```

**Evaluate:**
```bash
python eval.py \
  --predictions predictions.json \
  --gold validation_gold_schema_links.json \
  --schemas_dir schemas/ \
  --questions_input validation_input.json
```

### Optional flags

| Flag | Default | Description |
|---|---|---|
| `--adapter_dir` | `./adapter` | Path to LoRA adapter |
| `--base_model` | `Qwen/Qwen3-1.7B` | HuggingFace model ID (loaded from Hub) |
| `--schema_format` | `schema_sorted_pkfk` | Schema serialization format |
| `--schemas_dir` | `./schemas` | Path to schema JSON files |
| `--debug N` | `0` | Print raw model output for first N predictions |

---

## Model Artifact

The final trained model is a **LoRA adapter** stored in `./adapter/`.

- Base model: `Qwen/Qwen3-1.7B` (loaded automatically from HuggingFace Hub)
- Adapter: `./adapter/` (committed in this repo, ~50 MB)
- `main.py` loads the base model via `AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")`
  and attaches the adapter via `PeftModel.from_pretrained(base, "./adapter")`

**No manual setup needed** — the base model downloads automatically on first run.

---

## Dependencies

Install into a conda environment:

```bash
conda create -n cse234 python=3.11 -y
conda activate cse234

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers trl peft datasets accelerate bitsandbytes
```

`eval.py` has no third-party dependencies.

---

## Repository Structure

```
sft-experimentation/
├── main.py                          # ← TA grading entry point
├── eval.py                          # Evaluation script (table + column P/R/F1)
├── sql_to_schema_links.py           # Extracts schema_links from SQL (data augmentation)
│
├── adapter/                         # ← Final LoRA adapter (test20-a)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer_config.json (+ tokenizer files)
│
├── schemas/                         # 17 Spider-format database schemas
│   ├── NTSB.json
│   ├── NYSED_SRC2022.json
│   ├── SBODemoUS-Finance.json
│   └── ... (14 more)
│
├── train.json                       # 301 original training examples
├── augmented_train_v2.json          # 3000 paraphrased training examples (aug_v2)
├── validation_input.json            # 101 validation questions (input-only)
├── validation_gold_schema_links.json # Gold answers for validation
│
├── test20.py                        # Final training script (RTX PRO 6000, bf16)
├── test20b.py                       # Final training script variant (RTX 4070, QLoRA)
├── test19.py                        # Previous experiment (CED v1)
├── test18.py                        # Previous experiment (pkfk format baseline)
├── test17.py                        # Previous experiment (QLoRA, RTX 4070)
├── test16.py                        # Previous experiment (full bf16, 3 configs)
├── test[1-15].py                    # Earlier experimental scripts
│
├── augment.py                       # Training data augmentation (paraphrasing)
├── result.md                        # All experiment results log
│
└── adapters/                        # All trained adapter checkpoints
    ├── test16-a/  test16-b/  test16-c/
    ├── test18-a/  test18-b/
    ├── test19-a/  test19-b/
    ├── test20-a/                    # ← Best result (LB=0.4990), copied to adapter/
    └── test20-b/
```

---

## Final Pipeline

### Base Model
- **Qwen/Qwen3-1.7B** (1.7B parameters, instruction-tuned, `enable_thinking=False`)

### Schema Format: `pkfk`
Tables and columns sorted A→Z. Each column annotated with type and PK/FK flag:
```
Schema:
  CRASH(CASEID:int[PK], CASESTATE:text, NUMFATAL:int, ...)
  INJURY(AIS:int, CASEID:int[FK], REGION:text, ...)
```

### LoRA Configuration (test20-a, final)
| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Epochs | 4 |
| Learning rate | 2e-4 (cosine schedule) |
| Batch size | 2 × grad_accum 2 = effective 4 |
| Max sequence length | 2048 |
| Quantization | None (bf16, RTX PRO 6000) |

### Training Data
- **aug_v2** (2921 examples after filtering): `augmented_train_v2.json` paraphrased from
  301 original training questions, zero-column examples removed
- **CED v2** (~4000 examples): Coverage Extension Data generated for all tables in all
  17 validation databases, including 229 tables never seen in aug_v2
- **Combined**: ~7000 examples total

### Post-processing
1. Parse JSON from model output (with repair for truncated output)
2. Filter hallucinated tables/columns against the input schema (case-insensitive)
3. Deduplicate columns within each table

---

## Experiment Summary

| Method | Model | Config | LB Score |
|---|---|---|---|
| test16-a | Qwen3-1.7B | r=16, alpha=32, len=1024, 3ep | 0.3326 |
| test16-b | Qwen3-1.7B | r=16, alpha=32, len=1024, lr=2e-4 | 0.3986 |
| test16-c | Qwen3-1.7B | r=16, alpha=32, len=2048, dropout=0.2 | 0.3798 |
| test17-a | Qwen3-1.7B | QLoRA, alpha=24, lr=2e-4 | — |
| test17-b | Qwen3-1.7B | QLoRA, alpha=16, lr=1e-4 | — |
| test18-a | Qwen2.5-1.5B | pkfk, alpha=16, len=2048 | 0.3990 |
| test18-b | Qwen3-1.7B | pkfk, alpha=16, len=2048 | 0.4405 |
| test19-a | Qwen2.5-1.5B | pkfk, alpha=32, len=2048, CED v1 | 0.4427 |
| test19-b | Qwen3-1.7B | pkfk, alpha=16, len=2048, CED v1 | 0.4486 |
| **test20-a** | **Qwen3-1.7B** | **pkfk, alpha=32, 4ep, CED v2** | **0.4990** |
| test20-b | Qwen3-1.7B | r=32, alpha=32, 3ep, CED v2 (QLoRA) | 0.2454 |

### Key Findings
- `max_length=1024` caused ~40% empty predictions → switching to 2048 was critical
- `pkfk` schema format (PK/FK annotations) outperformed plain typed schema
- CED v1 only extended columns for known tables; CED v2 added 229 previously-unseen
  validation tables, directly fixing "model always predicts CRASH for all NTSB questions"
- test20-b collapsed (r=32 + QLoRA on RTX 4070 with lower lr may have underfit)

---

## Dataset

Built from [SNAILS](https://github.com/KyleLuoma/SNAILS) (Luoma & Kumar, SIGMOD 2025):
- 17 databases: national-parks biodiversity, NTSB crash statistics, NY state education,
  SAP business resource planning (9 sub-modules)
- 301 train / 101 validation / hidden test questions
- Ground truth extracted from gold SQL via `sql_to_schema_links.py`

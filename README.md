# CSE/DSC 234 Project 2 — Schema Linking with SFT

Final validation leaderboard score: **0.5134** (Table 0.6001, Column 0.4268).

## Run (TA grading command)

```bash
python3 main.py --input validation_input.json --output predictions.json
```

Defaults: `--adapter_dir ./adapter`, `--base_model Qwen/Qwen3-1.7B`,
`--schema_format schema_sorted_pkfk`, `--schemas_dir ./schemas`.
The base model downloads automatically from HuggingFace on first run.
Recall post-processing is on by default (`--no_postprocess` to disable).

## Evaluate

```bash
python eval.py --predictions predictions.json \
  --gold validation_gold_schema_links.json \
  --schemas_dir schemas/ --questions_input validation_input.json
```

## Final pipeline

Qwen3-1.7B + LoRA adapter (`test20-a`: r=16, alpha=32, 4 epochs, lr=2e-4,
max_length=2048, bf16, completion-only loss) trained on `aug_v2` (3k paraphrases)
+ CED-v2 (~4k coverage-extension examples covering all 17 validation schemas).
Inference: `pkfk` schema serialization, greedy decode, JSON parse/repair, filter
against schema, dedup, then recall post-processing (keyword-column + table
augmentation) which lifts 0.4990 → 0.5134 with no retraining.

## Layout

```
CSE234-Project2/
├── main.py                 # grading entry point (loads ./adapter)
├── eval.py                 # metric (table + column P/R/F1)
├── sql_to_schema_links.py  # gold-link extraction from SQL
├── report.tex / report.pdf # 3-page report (both experiment rounds)
├── predictions.json        # final validation output (score 0.5134)
├── validation_input.json, validation_gold_schema_links.json
├── adapter/                # final LoRA adapter (test20-a)
├── schemas/                # 17 SNAILS database schemas
├── data/                   # train.json (301) + augmented_train_*.json + augment scripts
├── experiments/            # exp1-5.py (Round 1, RapidFire AI) + test1-25.py (Round 2)
└── logs/                   # result.md, result_exp.md, test*-log.txt, rapidfire/
```

## Experiment rounds

- **Round 1** (`experiments/exp1-5.py`, RapidFire AI, scores in `logs/result_exp.md`):
  prompt/schema-serialization sweep on the **original 301 examples**. Best 0.2425
  (schema_sorted). Plateaued due to validation-table coverage gap.
- **Round 2** (`experiments/test1-25.py`, scores in `logs/result.md`): data
  augmentation + LoRA/optimization sweep. Best SFT 0.4990 (test20-a); final 0.5134
  with post-processing.

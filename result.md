# 🏆 Schema Linking Experiments & Hyperparameter Tuning Logs

## 📊 Summary Leaderboard

| Experiment ID | Core Strategy Summary | Leaderboard Score | Rank |
| :--- | :--- | :--- | :--- |
| **test1-3** | Qwen2.5-1.5B + aug_v2 + LoRA r=16 **(+ PK/FK hints)** | **0.4415** | 👑 1 |
| **test11** | max_length=2048, lora_alpha=16 (alpha==r strategy) | **0.4278** | 🥈 2 |
| **test7-1** | Qwen3-1.7B + PK/FK annotations | **0.3926** | 🥉 3 |
| **test1-1** | Qwen2.5-1.5B + aug_v2 + schema_sorted | **0.3884** | 4 |
| **test10** | completion_only_loss=False (Full-sequence training) | **0.3744** | 5 |
| **test5** | Qwen3-1.7B + aug_v3 (5,000 examples) | **0.3645** | 6 |
| **test9** | Two-Stage / Chain-of-Thought (Table -> Column) | **0.3638** | 7 |
| **test2-2** | SmolLM2-1.7B (Alternative architecture) | **0.3615** | 8 |
| **test6** | Qwen2.5-1.5B + aug_v4 + PK/FK | **0.3534** | 9 |
| **test2-1** | Qwen2.5-1.5B + QLoRA 4-bit r=32 | **0.3485** | 10 |
| **test7-3** | Qwen3-1.7B + warmup doubled to 0.10 | **0.3415** | 11 |
| **baseline** | Un-finetuned reference model | **0.3385** | 12 |
| **test1-2** | Qwen2.5-1.5B + PK/FK (Potential info overload) | **0.3380** | 13 |
| **test12-a** | Qwen3-1.7B + LoRA r=64, alpha=128 (High capacity) | **0.3357** | 14 |
| **test7-2** | Qwen3-1.7B + Learning Rate halved (1e-4) | **0.3223** | 15 |
| **test8** | schema_sorted_origcol (Original DB column order) | **0.3174** | 16 |
| **test2-3** | Qwen2.5-0.5B + 5 epochs (Tiny model) | **0.3105** | 17 |
| **test12-b** | Qwen3-1.7B + LoRA r=32, alpha=64 + 5 epochs | **0.2695** | 18 |

---

## 📝 Detailed Logs

### Method 0: baseline
* **Description:** Baseline experiment using the un-finetuned model. Serves as the initial reference point for performance comparison.
```text
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.3871
Recall_T    : 0.4752
F1_T        : 0.3647
Table Score : 0.4090     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2608
Recall_C    : 0.3227
F1_C        : 0.2204
Column Score: 0.2680     ((P+R+F1)/3)

==> Leaderboard Score : 0.3385   (0.5Table + 0.5Column)
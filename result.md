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

Method 1: test1-1
Description: v2_sorted | Qwen2.5-1.5B | schema_sorted | aug_v2 data | LoRA r=16

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4563
Recall_T    : 0.5586
F1_T        : 0.4284
Table Score : 0.4811     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3189
Recall_C    : 0.3219
F1_C        : 0.2464
Column Score: 0.2957     ((P+R+F1)/3)

==> Leaderboard Score : 0.3884   (0.5Table + 0.5Column)
Method 2: test1-2
Description: Same setup as Method 1 (Qwen2.5-1.5B-Instruct + v2 data) but introduced Primary Key (PK) and Foreign Key (FK) annotations to the prompt (schema_sorted_pkfk) to help the model understand table relationships.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.3959
Recall_T    : 0.4505
F1_T        : 0.3622
Table Score : 0.4029     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2965
Recall_C    : 0.2903
F1_C        : 0.2327
Column Score: 0.2732     ((P+R+F1)/3)

==> Leaderboard Score : 0.3380   (0.5Table + 0.5Column)
Method 3: test1-3
Description: v2_pkfk | Qwen2.5-1.5B | schema_sorted_pkfk | aug_v2 data | LoRA r=16 (+ PK/FK hints)

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.5244
Recall_T    : 0.6411
F1_T        : 0.4960
Table Score : 0.5538     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3102
Recall_C    : 0.4124
F1_C        : 0.2649
Column Score: 0.3292     ((P+R+F1)/3)

==> Leaderboard Score : 0.4415   (0.5Table + 0.5Column)
Method 4: test2-1
Description: qlora_r32 | Qwen2.5-1.5B | schema_sorted | QLoRA 4-bit r=32 (higher capacity, less VRAM)

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.3904
Recall_T    : 0.5025
F1_T        : 0.3682
Table Score : 0.4204     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2735
Recall_C    : 0.3260
F1_C        : 0.2306
Column Score: 0.2767     ((P+R+F1)/3)

==> Leaderboard Score : 0.3485   (0.5Table + 0.5Column)
Method 5: test2-2
Description: smollm | SmolLM2-1.7B | schema_sorted | LoRA r=16 (different architecture)

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.3106
Recall_T    : 0.7351
F1_T        : 0.3427
Table Score : 0.4628     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.1568
Recall_C    : 0.4520
F1_C        : 0.1716
Column Score: 0.2601     ((P+R+F1)/3)

==> Leaderboard Score : 0.3615   (0.5Table + 0.5Column)
Method 6: test2-3
Description: qwen05_5ep | Qwen2.5-0.5B | schema_sorted | LoRA r=32 5ep (small model, more epochs)

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.2870
Recall_T    : 0.5941
F1_T        : 0.3132
Table Score : 0.3981     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.1718
Recall_C    : 0.3380
F1_C        : 0.1591
Column Score: 0.2230     ((P+R+F1)/3)

==> Leaderboard Score : 0.3105   (0.5Table + 0.5Column)
Method 7: test5
Description: Qwen3-1.7B trained on augmented_train_v3.json (5,000 examples)

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4170
Recall_T    : 0.5776
F1_T        : 0.3977
Table Score : 0.4641     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2334
Recall_C    : 0.3565
F1_C        : 0.2048
Column Score: 0.2649     ((P+R+F1)/3)

==> Leaderboard Score : 0.3645   (0.5Table + 0.5Column)
Method 8: test6
Description: Trained Qwen2.5-1.5B-Instruct using the enhanced dataset (augmented_train_v4.json) combined with the schema format containing PK/FK annotations (schema_sorted_pkfk).

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4056
Recall_T    : 0.5066
F1_T        : 0.3770
Table Score : 0.4297     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2799
Recall_C    : 0.3165
F1_C        : 0.2346
Column Score: 0.2770     ((P+R+F1)/3)

==> Leaderboard Score : 0.3534   (0.5Table + 0.5Column)
Method 9: test7-1
Description: Hyperparameter sweep based on the top-performing Qwen3-1.7B. This qwen3_pkfk experiment tests whether adding PK/FK annotations to the best model yields further improvements.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4366
Recall_T    : 0.5668
F1_T        : 0.4272
Table Score : 0.4769     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3143
Recall_C    : 0.3386
F1_C        : 0.2719
Column Score: 0.3083     ((P+R+F1)/3)

==> Leaderboard Score : 0.3926   (0.5Table + 0.5Column)
Method 10: test7-2
Description: Hyperparameter sweep for Qwen3-1.7B (qwen3_lr1e4). Halved the default learning rate from 2e-4 to 1e-4 to observe if smaller step sizes improve convergence.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.3500
Recall_T    : 0.5281
F1_T        : 0.3461
Table Score : 0.4081     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2110
Recall_C    : 0.3117
F1_C        : 0.1867
Column Score: 0.2365     ((P+R+F1)/3)

==> Leaderboard Score : 0.3223   (0.5Table + 0.5Column)
Method 11: test7-3
Description: Final hyperparameter sweep for Qwen3-1.7B (qwen3_warm). Maintained the 2e-4 learning rate but doubled the warmup ratio from 0.05 to 0.10.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4328
Recall_T    : 0.4851
F1_T        : 0.3890
Table Score : 0.4356     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2584
Recall_C    : 0.2685
F1_C        : 0.2152
Column Score: 0.2474     ((P+R+F1)/3)

==> Leaderboard Score : 0.3415   (0.5Table + 0.5Column)
Method 12: test8
Description: A control test focusing on schema formatting (schema_sorted_origcol). While table names remain sorted A→Z, columns are kept in their original database definition order. This aims to verify if preserving the original semantic grouping of adjacent columns helps improve Column-level scores.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.3679
Recall_T    : 0.4860
F1_T        : 0.3657
Table Score : 0.4065     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2340
Recall_C    : 0.2652
F1_C        : 0.1859
Column Score: 0.2283     ((P+R+F1)/3)

==> Leaderboard Score : 0.3174   (0.5Table + 0.5Column)
Method 13: test9
Description: A targeted experiment addressing the bottleneck where Table Score is acceptable (0.55+) but Column Score is low (0.32+). Introduced a "Two-Stage / Chain-of-Thought" output format (schema_sorted_2stage). The model is forced to explicitly declare selected tables on the first line (e.g., Tables: ["Table1", "Table2"]) before predicting specific schema links on the second line. The hypothesis is that making the model "commit" semantically and narrow down the table scope prevents hallucination and disorientation among massive columns, significantly boosting Column Precision.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4571
Recall_T    : 0.4926
F1_T        : 0.4071
Table Score : 0.4523     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2733
Recall_C    : 0.3147
F1_C        : 0.2378
Column Score: 0.2753     ((P+R+F1)/3)

==> Leaderboard Score : 0.3638   (0.5Table + 0.5Column)
Method 14: test10
Description: Training on the full token sequence (prompt + completion) rather than only the completion tokens. This may improve generalization by providing the model with more gradient signals about how the schema relates to the question. Changed completion_only_loss=False from the Method 3 baseline (which uses True).

Plaintext
--- Table-level (macro-averaged across questions) ----
Precision_T : 0.4157
Recall_T    : 0.5784
F1_T        : 0.4041
Table Score : 0.4661     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2534
Recall_C    : 0.3616
F1_C        : 0.2333
Column Score: 0.2828     ((P+R+F1)/3)

==> Leaderboard Score : 0.3744   (0.5Table + 0.5Column)
Method 15: test11
Description: Increased max_length=2048 (baseline uses 1024) to fit longer schemas without truncation. Set lora_alpha=16 (baseline uses 32), adopting the standard "no scaling" configuration where alpha == r.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4687
Recall_T    : 0.5611
F1_T        : 0.4485
Table Score : 0.4928     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3580
Recall_C    : 0.4080
F1_C        : 0.3227
Column Score: 0.3629     ((P+R+F1)/3)

==> Leaderboard Score : 0.4278   (0.5Table + 0.5Column)
Method 16: test12-a
Description: Experiment with Qwen3-1.7B, schema_sorted format, and aug_v2 data. Configured with a higher LoRA capacity (r=64, alpha=128), learning rate of 2e-4, trained over 3 epochs.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4407
Recall_T    : 0.3688
F1_T        : 0.3786
Table Score : 0.3960     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3101
Recall_C    : 0.2598
F1_C        : 0.2564
Column Score: 0.2754     ((P+R+F1)/3)

==> Leaderboard Score : 0.3357   (0.5Table + 0.5Column)
Method 17: test12-b
Description: Experiment with Qwen3-1.7B, schema_sorted format, and aug_v2 data. Configured with a moderate LoRA capacity (r=32, alpha=64), learning rate of 2e-4, trained for an extended 5 epochs.

Plaintext
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.3324
Recall_T    : 0.2970
F1_T        : 0.2901
Table Score : 0.3065     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2619
Recall_C    : 0.2196
F1_C        : 0.2159
Column Score: 0.2325     ((P+R+F1)/3)

==> Leaderboard Score : 0.2695   (0.5Table + 0.5Column)
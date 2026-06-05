# 🏆 Schema Linking Experiments & Hyperparameter Tuning Logs

## 📊 Summary Leaderboard

| Experiment ID | Core Strategy Summary | Leaderboard Score | Rank |
| :--- | :--- | :--- | :--- |
| **test19-b** | Qwen3-1.7B + pkfk + alpha=16 + len=2048 + aug_v2+CED | **0.4486** | 👑 1 (Overall Winner) |
| **test19-a** | Qwen2.5-1.5B + pkfk + alpha=32 + len=2048 + aug_v2+CED | **0.4427** | 2 |
| **test1-3** | Qwen2.5-1.5B + aug_v2 + LoRA r=16 (+ PK/FK hints) | **0.4415** | 3 |
| **test18-b** | Qwen3-1.7B + pkfk + alpha=16 + len=2048 | **0.4405** | 4 |
| **test11** | max_length=2048, lora_alpha=16 (alpha==r strategy) | **0.4278** | 5 |
| **test14-a** | Qwen3-1.7B + QLoRA + r=16 + len=2048 | **0.4012** | 6 |
| **test14-b** | Qwen3-1.7B + QLoRA + r=16 + alpha=32 | **0.3997** | 7 |
| **test18-a** | Qwen2.5-1.5B + pkfk + alpha=16 + len=2048 | **0.3990** | 8 |
| **test7-1** | Qwen3-1.7B + PK/FK annotations | **0.3926** | 9 |
| **test1-1** | Qwen2.5-1.5B + aug_v2 + schema_sorted | **0.3884** | 10 |
| **test16-c** | lr=5e-5 + dropout=0.20 + len=2048 | **0.3798** | 11 |
| **test10** | completion_only_loss=False (Full-sequence training) | **0.3744** | 12 |
| **test5** | Qwen3-1.7B + aug_v3 (5,000 examples) | **0.3645** | 13 |
| **test9** | Two-Stage / Chain-of-Thought (Table -> Column) | **0.3638** | 14 |
| **test2-2** | SmolLM2-1.7B (Alternative architecture) | **0.3615** | 15 |
| **test6** | Qwen2.5-1.5B + aug_v4 + PK/FK | **0.3534** | 16 |
| **test2-1** | Qwen2.5-1.5B + QLoRA 4-bit r=32 | **0.3485** | 17 |
| **test7-3** | Qwen3-1.7B + warmup doubled to 0.10 | **0.3415** | 18 |
| **baseline** | Un-finetuned reference model | **0.3385** | 19 |
| **test1-2** | Qwen2.5-1.5B + PK/FK (Potential info overload) | **0.3380** | 20 |
| **test12-a** | Qwen3-1.7B + LoRA r=64, alpha=128 (High capacity) | **0.3357** | 21 |
| **test7-2** | Qwen3-1.7B + Learning Rate halved (1e-4) | **0.3223** | 22 |
| **test8** | schema_sorted_origcol (Original DB column order) | **0.3174** | 23 |
| **test2-3** | Qwen2.5-0.5B + 5 epochs (Tiny model) | **0.3105** | 24 |
| **test16-a** | lr=5e-5 + dropout=0.10 + len=1024 | **0.2507** | 25 |
| **test12-b** | Qwen3-1.7B + LoRA r=32, alpha=64 + 5 epochs | **0.2695** | 26 |
| **test16-b** | lr=2e-4 + dropout=0.10 + len=1024 | **0.2142** | 27 |

***Note:***
* 👑 **Overall Leader:** `test19-b` (0.4486)
* 🎯 **Table-level Record:** `test1-3` (0.5538)
* 🧩 **Column-level Record:** `test19-b` (0.3791)

---

---

## 📝 Detailed Logs

### Method 0: baseline
* **Description:** Baseline experiment using the un-finetuned model. Serves as the initial reference point for performance comparison.
```text
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.3871
Recall_T    : 0.4752
F1_T        : 0.3647
 : 0.4090     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.2608
Recall_C    : 0.3227
F1_C        : 0.2204
Column Score: 0.2680     ((P+R+F1)/3)

==> Leaderboard Score : 0.3385   (0.5Table + 0.5Column)

Method 1: test1-1
Description: v2_sorted | Qwen2.5-1.5B | schema_sorted | aug_v2 data | LoRA r=16


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

Method 18: test13-a
Description: Qwen3-1.7B  schema_sorted  aug_v2  r=16 alpha=16  max_length=2048  3 epochs
Method 19: test13-b
escription: identical to test13-a but 4 epochs

Method 20: test14-a
Description: Qwen3-1.7B QLoRA  schema_sorted  aug_v2  r=16 alpha=16  2048  3ep
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.5176
Recall_T    : 0.4645
F1_T        : 0.4626
Table Score : 0.4816     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3474
Recall_C    : 0.3218
F1_C        : 0.2935
Column Score: 0.3209     ((P+R+F1)/3)

==> Leaderboard Score : 0.4012   (0.5*Table + 0.5*Column)

Method 21: test14-b
Description: identical to test14-a but alpha=32  (isolates alpha effect)
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.4587
Recall_T    : 0.5083
F1_T        : 0.4418
Table Score : 0.4696     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3328
Recall_C    : 0.3567
F1_C        : 0.3001
Column Score: 0.3299     ((P+R+F1)/3)

==> Leaderboard Score : 0.3997   (0.5*Table + 0.5*Column)

Method 22: test15-a
Method 23: test15-b
Method 24: test15-b

Method 25: test16-a
Descriotion: lr=5e-5  dropout=0.10  epochs=3  len=1024  (slow lr + mild dropout)
---- Table-level (macro-averaged across questions) ----                                                                                        
Precision_T : 0.3026                                                                                                                         
Recall_T    : 0.3515                                                                                                                         
F1_T        : 0.2877                                                                                                                         
Table Score : 0.3139     ((P+R+F1)/3)                                                                                                        
                                                                                                                                            
---- Column-level (Table.Column pairs, macro-averaged) ----                                                                                    
Precision_C : 0.2086                                                                                                                         
Recall_C    : 0.1834
F1_C        : 0.1704
Column Score: 0.1875     ((P+R+F1)/3)

==> Leaderboard Score : 0.2507   (0.5*Table + 0.5*Column)

Method 26: test16-b  
Descriotion: lr=2e-4  dropout=0.10  epochs=2  len=1024  (early stop before overfit)
  
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.2436
Recall_T    : 0.2805
F1_T        : 0.2213
Table Score : 0.2485     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.1949
Recall_C    : 0.1841
F1_C        : 0.1608
Column Score: 0.1799     ((P+R+F1)/3)

==> Leaderboard Score : 0.2142   (0.5*Table + 0.5*Column)

Method 27: test16-c  
Descriotion: lr=5e-5  dropout=0.20  epochs=3  len=2048  (strong reg + long ctx)
--- Table-level (macro-averaged across questions) ----
Precision_T : 0.4540
Recall_T    : 0.4827
F1_T        : 0.4257
Table Score : 0.4541     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3435
Recall_C    : 0.3016
F1_C        : 0.2713
Column Score: 0.3055     ((P+R+F1)/3)

==> Leaderboard Score : 0.3798   (0.5*Table + 0.5*Column)

Method 28: test17-a  
Descriotion: alpha=24  lr=2e-4   (intermediate alpha to balance P and R)
  
Method 29: test17-b  
Descriotion: alpha=16  lr=1e-4   (alpha=16 kept; lower lr to recover recall)

Method 30: test18-a
Descriotion: Qwen2.5-1.5B-Instruct + pkfk + alpha=16 + max_length=2048
            = test1-3's winning model & format  +  test11's alpha/context
            Hypothesis: best table recall from pkfk × best column score from alpha=16

---- Table-level (macro-averaged across questions) ----
Precision_T : 0.5173
Recall_T    : 0.4835
F1_T        : 0.4751
Table Score : 0.4919     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3643
Recall_C    : 0.2723
F1_C        : 0.2818
Column Score: 0.3061     ((P+R+F1)/3)

==> Leaderboard Score : 0.3990   (0.5*Table + 0.5*Column)

Method 31: test18-b
Descriotion: Qwen3-1.7B + pkfk + alpha=16 + max_length=2048
        = test7-1's pkfk Qwen3 config  +  test11's alpha/context
        Hypothesis: Qwen3's stronger base + pkfk + alpha=16 combo

---- Table-level (macro-averaged across questions) ----
Precision_T : 0.5100
Recall_T    : 0.5751
F1_T        : 0.4980
Table Score : 0.5277     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3619
Recall_C    : 0.3709
F1_C        : 0.3274
Column Score: 0.3534     ((P+R+F1)/3)

==> Leaderboard Score : 0.4405   (0.5*Table + 0.5*Column)

Method 32: test19-a
Description: Qwen2.5-1.5B-Instruct  pkfk  alpha=32  len=2048  aug_v2+CED
---- Table-level (macro-averaged across questions) ----
Precision_T : 0.5330
Recall_T    : 0.5272
F1_T        : 0.5046
Table Score : 0.5216     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3994
Recall_C    : 0.3536
F1_C        : 0.3384
Column Score: 0.3638     ((P+R+F1)/3)

==> Leaderboard Score : 0.4427   (0.5*Table + 0.5*Column)

Method 33: test19-b
Description: Qwen3-1.7B              pkfk  alpha=16  len=2048  aug_v2+CED
---- Table-level (macro-
averaged across questions) ----
Precision_T : 0.5176
Recall_T    : 0.5413
F1_T        : 0.4955
Table Score : 0.5181     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
Precision_C : 0.3832
Recall_C    : 0.4013
F1_C        : 0.3527
Column Score: 0.3791     ((P+R+F1)/3)

==> Leaderboard Score : 0.4486   (0.5*Table + 0.5*Column)

Method 34: test20-a
Description: Qwen3-1.7B  pkfk  r=16  alpha=32  CED-v2  4 epochs  lr=2e-4
---- Table-level (macro-averaged across questions) ----
  Precision_T : 0.5675
  Recall_T    : 0.6444
  F1_T        : 0.5647
  Table Score : 0.5922     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
  Precision_C : 0.4015
  Recall_C    : 0.4430
  F1_C        : 0.3727
  Column Score: 0.4057     ((P+R+F1)/3)

==> Leaderboard Score : 0.4990   (0.5*Table + 0.5*Column)

Method 35: test20-b
Description: r=32 doubles LoRA rank, potentially better for the large
  new schema space; 3 epochs with lower lr avoids overfitting
---- Table-level (macro-averaged across questions) ----
  Precision_T : 0.2594
  Recall_T    : 0.2731
  F1_T        : 0.2394
  Table Score : 0.2573     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
  Precision_C : 0.2789
  Recall_C    : 0.2117
  F1_C        : 0.2100
  Column Score: 0.2335     ((P+R+F1)/3)

==> Leaderboard Score : 0.2454   (0.5*Table + 0.5*Column)

Method 36: test21
---- Table-level (macro-averaged across questions) ----
  Precision_T : 0.4989
  Recall_T    : 0.5611
  F1_T        : 0.4876
  Table Score : 0.5158     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
  Precision_C : 0.3668
  Recall_C    : 0.3982
  F1_C        : 0.3368
  Column Score: 0.3673     ((P+R+F1)/3)

==> Leaderboard Score : 0.4416   (0.5*Table + 0.5*Column)

Method 37: test22
Description:
---- Table-level (macro-averaged across questions) ----
  Precision_T : 0.4683
  Recall_T    : 0.5809
  F1_T        : 0.4728
  Table Score : 0.5073     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
  Precision_C : 0.3590
  Recall_C    : 0.4339
  F1_C        : 0.3514
  Column Score: 0.3814     ((P+R+F1)/3)

==> Leaderboard Score : 0.4444   (0.5*Table + 0.5*Column)

Method 38: test23
---- Table-level (macro-averaged across questions) ----
  Precision_T : 0.5284
  Recall_T    : 0.5338
  F1_T        : 0.5050
  Table Score : 0.5224     ((P+R+F1)/3)

---- Column-level (Table.Column pairs, macro-averaged) ----
  Precision_C : 0.4239
  Recall_C    : 0.4013
  F1_C        : 0.3824
  Column Score: 0.4025     ((P+R+F1)/3)

==> Leaderboard Score : 0.4625   (0.5*Table + 0.5*Column)
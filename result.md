### Method 0: baseline
* **說明**：
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

==> Leaderboard Score : 0.3385   (0.5*Table + 0.5*Column)

### Method 1: test1-v2-sorter
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

==> Leaderboard Score : 0.3884   (0.5*Table + 0.5*Column)
### Method 2: test1-v2-pkfk
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

==> Leaderboard Score : 0.3380   (0.5*Table + 0.5*Column)
### Method 3: test1-qwen3
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

==> Leaderboard Score : 0.4415   (0.5*Table + 0.5*Column)
### Method 4: test2-qlora-r32

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

==> Leaderboard Score : 0.3485   (0.5*Table + 0.5*Column)

### Method 5: test2-smollm
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

==> Leaderboard Score : 0.3615   (0.5*Table + 0.5*Column)

### Method 6: test2-qwen05-5ep
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

==> Leaderboard Score : 0.3105   (0.5*Table + 0.5*Column)
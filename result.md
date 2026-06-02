### Method 0: Baseline
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
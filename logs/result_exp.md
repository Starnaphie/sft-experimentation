# Schema Linking 實驗紀錄

## 📊 實驗結果總表 (Leaderboard Score Comparison)

| Metrics | Baseline | Exp1: Fewshot | Exp1: Typed | Exp1: Fewshot-Typed | Exp2: Schema-Abbrev | Exp2: Schema-Sorted | Exp2: Schema-Top 10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Table Precision** | 0.0027 | 0.0006 | 0.2378 | 0.1564 | 0.2473 | 0.2705 | 0.0074 |
| **Table Recall** | 0.0248 | 0.0099 | 0.3350 | 0.2954 | 0.3432 | 0.4035 | 0.0347 |
| **Table F1** | 0.0048 | 0.0012 | 0.2368 | 0.1682 | 0.2544 | 0.2770 | 0.0118 |
| **Table Score** | 0.0108 | 0.0039 | 0.2698 | 0.2067 | 0.2816 | 0.3170 | 0.0179 |
| **Column Precision**| 0.0396 | 0.0297 | 0.2047 | 0.1730 | 0.2058 | 0.1913 | 0.0396 |
| **Column Recall** | 0.0347 | 0.0297 | 0.1897 | 0.1334 | 0.1866 | 0.1579 | 0.0356 |
| **Column F1** | 0.0363 | 0.0297 | 0.1708 | 0.1333 | 0.1769 | 0.1548 | 0.0371 |
| **Column Score** | 0.0369 | 0.0297 | 0.1884 | 0.1466 | 0.1898 | 0.1680 | 0.0375 |
| **🏆 Leaderboard Score**| **0.0238** | **0.0168** | **0.2291** | **0.1766** | **0.2357** | **0.2425** | **0.0277** |

*註：Table Score 與 Column Score 為 (P+R+F1)/3；Leaderboard Score 為 (0.5 * Table + 0.5 * Column)。*

---
## 📝 實驗方法與詳細數據紀錄

### Method 1: Baseline
* **說明**：未經過 Prompt 優化的初始基準點測試。
* **Table-level (Macro-averaged)**
  * Precision: 0.0027
  * Recall: 0.0248
  * F1: 0.0048
  * **Table Score: 0.0108**
* **Column-level (Macro-averaged)**
  * Precision: 0.0396
  * Recall: 0.0347
  * F1: 0.0363
  * **Column Score: 0.0369**
* **🏆 Leaderboard Score: 0.0238**

### Method 2: Exp1-Fewshot
* **說明**：使用精簡版 Schema (Compact schema) 並在 Prompt 中加入 One-shot/Few-shot 範例 (in-context example)。
* **Table-level (Macro-averaged)**
  * Precision: 0.0006
  * Recall: 0.0099
  * F1: 0.0012
  * **Table Score: 0.0039**
* **Column-level (Macro-averaged)**
  * Precision: 0.0297
  * Recall: 0.0297
  * F1: 0.0297
  * **Column Score: 0.0297**
* **🏆 Leaderboard Score: 0.0168**

### Method 3: Exp1-Typed
* **說明**：使用精簡版 Schema，並在 Schema 中明確加入資料型態 (type annotations) 標註。
* **Table-level (Macro-averaged)**
  * Precision: 0.2378
  * Recall: 0.3350
  * F1: 0.2368
  * **Table Score: 0.2698**
* **Column-level (Macro-averaged)**
  * Precision: 0.2047
  * Recall: 0.1897
  * F1: 0.1708
  * **Column Score: 0.1884**
* **🏆 Leaderboard Score: 0.2291**

### Method 4: Exp1-Fewshot-Typed 
* **說明**：結合上述兩者，使用帶有資料型態 (type annotations) 的 Schema，同時加入 in-context example 範例。
* **Table-level (Macro-averaged)**
  * Precision: 0.1564
  * Recall: 0.2954
  * F1: 0.1682
  * **Table Score: 0.2067**
* **Column-level (Macro-averaged)**
  * Precision: 0.1730
  * Recall: 0.1334
  * F1: 0.1333
  * **Column Score: 0.1466**
* **🏆 Leaderboard Score: 0.1766**

### Method 5: Exp2-Schema-Abbrev
* **說明**：把 column type 縮寫（text→T, number→N, real→R, time→TM），減少 token 讓模型更專注在名稱上
* **Table-level (Macro-averaged)**
  * Precision: 0.2473
  * Recall: 0.3432
  * F1: 0.2544
  * **Table Score: 0.2816**
* **Column-level (Macro-averaged)**
  * Precision: 0.2058
  * Recall: 0.1866
  * F1: 0.1769
  * **Column Score: 0.1898**
* **🏆 Leaderboard Score: 0.2357**

### Method 6: Exp2-Schema-Sorted
* **說明**：把 table/column 按名稱字母排序，給模型更一致的輸入結構
* **Table-level (Macro-averaged)**
  * Precision: 0.2705
  * Recall: 0.4035
  * F1: 0.2770
  * **Table Score: 0.3170**
* **Column-level (Macro-averaged)**
  * Precision: 0.1913
  * Recall: 0.1579
  * F1: 0.1548
  * **Column Score: 0.1680**
* **🏆 Leaderboard Score: 0.2425**

### Method 7: Exp2-Schema-Top10
* **說明**：用 keyword matching 預先過濾，每個 question 只給最相關的前 10 個 table，減少噪音
* **Table-level (Macro-averaged)**
  * Precision: 0.0074
  * Recall: 0.0347
  * F1: 0.0118
  * **Table Score: 0.0179**
* **Column-level (Macro-averaged)**
  * Precision: 0.0396
  * Recall: 0.0356
  * F1: 0.0371
  * **Column Score: 0.0375**
* **🏆 Leaderboard Score: 0.0277**

### Method 8: Exp3-Sorted-Abbrev
* **說明**：結合 sorted + 縮寫 type，sorted 提升 table recall，abbrev 減少 token，看能不能同時拉高 table 和 column
* **Table-level (Macro-averaged)**
  * Precision: 0.1604
  * Recall: 0.3416
  * F1: 0.1824
  * **Table Score: 0.2281**
* **Column-level (Macro-averaged)**
  * Precision: 0.1356
  * Recall: 0.1085
  * F1: 0.1070
  * **Column Score: 0.1170**
* **🏆 Leaderboard Score: 0.1726**

### Method 9: Exp3-Question-Hint
* **說明**：在 schema 後面加一行 Key terms: {關鍵詞}，從 question 抽出名詞提示模型關注哪些 table
* **Table-level (Macro-averaged)**
  * Precision: 0.1407
  * Recall: 0.2616
  * F1: 0.1513
  * **Table Score: 0.1845**
* **Column-level (Macro-averaged)**
  * Precision: 0.0877
  * Recall: 0.0734
  * F1: 0.0673
  * **Column Score: 0.0761**
* **🏆 Leaderboard Score: 0.1303**

### Method 10: Exp3-Col-Filtered
* **說明**：每個 table 只保留跟 question keyword 有關的 column（最多保留前 8 個），減少 column 層面的噪音，針對性解決 column precision 偏低的問題
* **Table-level (Macro-averaged)**
  * Precision: 0.0013
  * Recall: 0.0099
  * F1: 0.0023
  * **Table Score: 0.0045**
* **Column-level (Macro-averaged)**
  * Precision: 0.0297
  * Recall: 0.0297
  * F1: 0.0297
  * **Column Score: 0.0297**
* **🏆 Leaderboard Score: 0.0171**

### Method 11: Exp4-Sorted-Typed-V2 
* **說明**：用 schema_sorted 的格式（目前最強），但把 epoch 從 3 增加到 5，看模型是否需要更多訓練才能收斂
* **Table-level (Macro-averaged)**
  * Precision: 0.2498
  * Recall: 0.3036
  * F1: 0.2274
  * **Table Score: 0.2603**
* **Column-level (Macro-averaged)**
  * Precision: 0.2320
  * Recall: 0.1565
  * F1: 0.1633
  * **Column Score: 0.1839**
* **🏆 Leaderboard Score: 0.2221**

### Method 12: Exp4-Col-Hint-Output
* **說明**：在 completion 裡，gold answer 之前加一行 Tables: [t1, t2] 讓模型先預測 table 再預測 column，引導模型分兩步思考
* **Table-level (Macro-averaged)**
  * Precision: 0.0151
  * Recall: 0.0396
  * F1: 0.0199
  * **Table Score: 0.0249**
* **Column-level (Macro-averaged)**
  * Precision: 0.0426
  * Recall: 0.0386
  * F1: 0.0402
  * **Column Score: 0.0405**
* **🏆 Leaderboard Score: 0.0327**

### Method 13: Exp4-Schema-Sorted-Desc
* **說明**： sorted 格式但 table 按字母倒序排列，測試模型是否對排序方向有偏好（有些研究發現 LLM 對 list 末尾更敏感）
* **Table-level (Macro-averaged)**
  * Precision: 0.2530
  * Recall: 0.3193
  * F1: 0.2469
  * **Table Score: 0.2731**
* **Column-level (Macro-averaged)**
  * Precision: 0.1675
  * Recall: 0.1186
  * F1: 0.1323
  * **Column Score: 0.1395**
* **🏆 Leaderboard Score: 0.2063**

### Method 14: Exp5-Preds-Sorted-Table-Orig-Col
* **說明**： table 按字母排序（保留 sorted 的 table 優勢），但 column 維持原始順序（保留 typed 的 column 優勢）
* **Table-level (Macro-averaged)**
  * Precision: 0.1886
  * Recall: 0.2838
  * F1: 0.1910
  * **Table Score: 0.2211**
* **Column-level (Macro-averaged)**
  * Precision: 0.2006
  * Recall: 0.1580
  * F1: 0.1561
  * **Column Score: 0.1716**
* **🏆 Leaderboard Score: 0.1963**

### Method 15: Exp5-Schema-Sorted-Lr
* **說明**： schema_sorted 格式但 learning rate 從 2e-4 降到 5e-5，看是否能在更精細的 gradient 下同時提升 table 和 column
* **Table-level (Macro-averaged)**
  * Precision: 0.2159
  * Recall: 0.3193
  * F1: 0.2147
  * **Table Score: 0.2500**
* **Column-level (Macro-averaged)**
  * Precision: 0.1659
  * Recall: 0.1240
  * F1: 0.1245
  * **Column Score: 0.1381**
* **🏆 Leaderboard Score: 0.1941**

### Method 16: Exp5-Typed-Augmented
* **說明**： typed 格式（column score 較好），但把每個訓練樣本的 schema 隨機 shuffle table 順序做 data augmentation，讓模型對順序不敏感、更能依靠語義
* **Table-level (Macro-averaged)**
  * Precision: 0.2171
  * Recall: 0.3564
  * F1: 0.2287
  * **Table Score: 0.2674**
* **Column-level (Macro-averaged)**
  * Precision: 0.1782
  * Recall: 0.1414
  * F1: 0.1511
  * **Column Score: 0.1569**
* **🏆 Leaderboard Score: 0.2122**

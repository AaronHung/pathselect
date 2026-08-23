# 專案憲法 — 不隨單次實驗改動的規則

這裡放**跨實驗、跨報告都適用**的政策。單次裁定寫在 `ledger/DR-0xx.md`；
會反覆套用的規則升格到這裡，並在對應的 DR 卡留下來源。

> 入口順序：`ledger/INDEX.md` → 本檔 → status=ACTIVE 的 DR 卡。

---

## §1 統計政策

### §1.1 三級 win count 規則（來源 [DR-020](ledger/DR-020.md)）

臂間比較一律**配對**（同 seed 相減）。win count 的判讀只有三級，全文統一用詞：

| win count | 名稱 | 判讀 |
|---|---|---|
| 5/5 | **systematic** | 系統性差異 |
| 4/5 | **directional, inconclusive** | 方向一致但證據不足以定案 |
| ≤3/5 | **within noise** | 落在雜訊內 |

### §1.2 n < 5 的批次（來源 [DR-023](ledger/DR-023.md)）

**三級規則是為 n = 5 校準的。** n < 5 的批次一律加註：

> 本批的 systematic 應讀作**方向一致**而非**已定案**；
> 寫進論文的主張必須回到 5-seed 的批次確認。

報告產生器（`scripts/run_exp2.py::write_paired`）在 `len(seeds) < 5` 時**自動加註**，
不依賴人工記得。

### §1.3 不報 p 值（來源 [DR-016](ledger/DR-016.md)）

n = 5 的樣本數下顯著性檢定會誤導。一律報逐 seed 配對差值、配對 mean ± std
與 win count。

### §1.4 效果量的最小可辨識度

esca 只有 15 張 test slide，**一張 = 6.67 pp**。esca 上小於該值的差異一律標註為
**不可區分**，不得用來支撐任何主張。

---

## §2 證據政策

### §2.1 逐 slide 存檔（來源 [DR-012](ledger/DR-012.md)、裁定 C）

`outputs/` 下每一次評估都必須落一份逐 slide JSON
（`slide_id` / `task` / `true` / `pred*` / 選取類另加 `selected_idx` / `weights`）。
由 `tests/test_per_slide_records.py` 把關。理由：v9 只存彙總 accuracy，
導致 flip 分析只能給區間。

### §2.2 科學主張要 mutation check（來源 `tests/README.md` §1b）

要宣稱「某模組有作用 / 無作用」時，優先用**替換或擾動實測**，不要只讀 code 推論；
並且必須做反向對照，否則分不清「模組沒作用」與「替換沒生效」。

### §2.3 斷言要 mutation check（來源 `tests/README.md` §1）

2026-08-24 之後新增的每條 assert，提交前必須以人工製造的違例確認它會失敗。
提交訊息以一行註明違例內容。

---

## §3 報告政策

### §3.1 forgetting 類指標只算前 T−1 個 task

最後學的 task 兩個時點是同一個，A1 恆為 0、Jaccard 恆為 1，算進去會稀釋量級。
final average accuracy 與洩漏率仍算全部 T 個。

### §3.2 負面結果照實報

方法沒贏 baseline、順序依賴、中段證據弱 —— 一律獨立成節寫明，不美化、
不調參數搶救。這是 [DR-015](ledger/DR-015.md) 與 [DR-019](ledger/DR-019.md) 的共同要求。

### §3.3 參照臂不是 baseline

per-task specialist（獨立訓練）與 joint offline 的 forgetting 由構造為 0，
不得用來宣稱 forgetting（[DR-011](ledger/DR-011.md)）。

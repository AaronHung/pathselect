# Decision Ledger — INDEX（L1 現況表）

> 未來重啟分析時的第一句話：
> 「先讀 `docs/ledger/INDEX.md` 與 `GRAVEYARD.md`，再讀 status=ACTIVE 的 DR 卡。
> L4 原始對話只在需要考古時進入。」

**規則 3 —— 狀態只有三種**：`ACTIVE` / `SUPERSEDED-BY DR-0xx` / `PARKED`。
`tests/test_ledger.py` 強制執行這張表的完整性與狀態合法性。

| DR | 標題 | status | 一句話 |
|---|---|---|---|
| [001](DR-001.md) | 雙軌（ICASSP+老師） | SUPERSEDED-BY DR-003 | v0.5 的 Track A/B 分軌 |
| [002](DR-002.md) | 遷移至 pathselect，舊 repo 唯讀封存 | ACTIVE | 6 檔搬移 + SHA256 凍結 + 禁字紅線測試 |
| [003](DR-003.md) | 單一主線 v0.8，freeze 9/2 → 論文 8/31 | ACTIVE | Sol 合併案；specialist / joint / SeqFT / CL 四位階 |
| [004](DR-004.md) | 用字禁令 | ACTIVE | Zero/Router 禁用；Navigation 保留給獨立 coarse view |
| [005](DR-005.md) | 操作點 B=8, c=1 | ACTIVE | budget 曲線峰值在 K=8（D1）；c=1 使 e_t 槓桿最大 |
| [006](DR-006.md) | 聚合權重：softmax 主線、等權=selection-only | ACTIVE | 訓練一致性原則；D2 eff_K 證據；「非依數值選定」已記錄 |
| [007](DR-007.md) | L_sem 主線 = discriminative prior | ACTIVE | max-sim 即 simple similarity（老師批評點）；與 v9 entropy 輸入連續 |
| [008](DR-008.md) | q_τ 定義凍結（class-text 平均） | ACTIVE | 定義 ACTIVE、不可學習化、leakage 雙 assert；**主張已 PARK → G-05** |
| [009](DR-009.md) | --group-grad 預設 ste_allocation | ACTIVE | none 下 F_g 無梯度 → hierarchy ablation 構造性 null |
| [010](DR-010.md) | L4-L6 與所有 q ablation 一律 joint 訓練 | ACTIVE | per-task 下 q 為常數被 bias 吸收 |
| [011](DR-011.md) | specialist ≠ oracle；joint = offline reference；遺忘只由 SeqFT 證明 | ACTIVE | Sol 紅隊裁定；跨任務矩陣只證 specialization |
| [012](DR-012.md) | 遺忘三軸 + task-IL/class-IL/洩漏率並報 | ACTIVE | 洩漏 100% 歸因選取（frozen head）→ 寫成 contribution |
| [013](DR-013.md) | replay=資料機制；L_replay := L_diag on M | ACTIVE | 圖（三項）與文（兩項）的矛盾以此定義收斂 |
| [014](DR-014.md) | beta_u=0.1 全臂保留；S2 降級 preliminary | ACTIVE | A1=最終 within-task 目標下的 SeqFT，才是隔離 CL 的正確 baseline |
| [015](DR-015.md) | 主張定調：replay→accuracy、KD+eq→behaviour | ACTIVE | task-IL +0.74pp 在雜訊內，不得宣稱 |
| [016](DR-016.md) | 配對統計 + win count，不報 p 值 | ACTIVE | 七臂同 seeds；n=5 政策沿用 |
| [017](DR-017.md) | E1 記憶體曲線為主要防禦；牌一先實作備援 | ACTIVE | ⏳ 執行中；防「把 \|M\| 開大就好」；utility-gated KD 在手 |
| [020](DR-020.md) | win count 三級規則與 A3 非單調的定性 | ACTIVE | 5/5 systematic、4/5 directional inconclusive、≤3/5 within noise；2× 為跨容量陳述 |

> DR-018 / DR-019 尚未配發，編號留空不補號（append-only，編號一旦跳過就不回填）。

墓園（柴火區）見 [GRAVEYARD.md](GRAVEYARD.md)；未解釋的現象見 [SEEDS.md](SEEDS.md)。

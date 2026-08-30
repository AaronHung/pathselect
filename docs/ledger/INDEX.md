# Decision Ledger — INDEX（L1 現況表）

> 未來重啟分析時的第一句話：
> 「先讀 `docs/ledger/INDEX.md` 與 [`docs/CONSTITUTION.md`](../CONSTITUTION.md)，
> 再讀 [`docs/CLAIMS.md`](../CLAIMS.md)、`GRAVEYARD.md` 與 status=ACTIVE 的 DR 卡。
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
| [015](DR-015.md) | 主張定調：replay→accuracy、KD+eq→behaviour | SUPERSEDED-BY DR-029 | flat 架構下的定調；階層版由 DR-029 解禁 |
| [016](DR-016.md) | 配對統計 + win count，不報 p 值 | ACTIVE | 七臂同 seeds；n=5 政策沿用 |
| [017](DR-017.md) | E1 記憶體曲線為主要防禦；牌一先實作備援 | ACTIVE | ⏳ 執行中；防「把 \|M\| 開大就好」；utility-gated KD 在手 |
| [018](DR-018.md) | 寫作策略：完整版母本優先，venue 視圖後導出 | ACTIVE | 不為任何 venue 砍 idea；去風險由 checklist/ledger/SEEDS 承擔 |
| [019](DR-019.md) | E1 判讀：防禦成立，記憶體效率主張定為 2× | SUPERSEDED-BY DR-042 | 四條可宣稱；中段 128/256 的弱證據須揭露；牌一改為稀缺端假設驅動 |
| [020](DR-020.md) | win count 三級規則與 A3 非單調的定性 | ACTIVE | 5/5 systematic、4/5 directional inconclusive、≤3/5 within noise；2× 為跨容量陳述 |
| [021](DR-021.md) | 架構一致性：CL 主線改用階層選取器 | ACTIVE | 主表 flat 與架構圖 Group→Patch 脫節；G1 判準已 pre-register |
| [022](DR-022.md) | group-level distillation 從未被測試；G1 增臂隔離 | SUPERSEDED-BY DR-035 | 退化階層下的「未顯示效果」結論已作廢 |
| [023](DR-023.md) | n<5 批次的統計警語升格為憲法條文 | ACTIVE | 規則本體移入 docs/CONSTITUTION.md §1.2；報告產生器自動加註 |
| [024](DR-024.md) | 跨批次配對與撤回程序入憲 | ACTIVE | 憲法 §1.3 共同子集不代表母體、§2.4 撤回不算失分隱瞞才算 |
| [025](DR-025.md) | 配額口徑改為對整個 budget；chunked loop 在無 state 下為 no-op | ACTIVE | per-chunk 在 c=1 必然退化；DR-021 判準沿用不改；新增 CLAIMS.md |
| [026](DR-026.md) | 執行期檔案凍結；長 job 必須檢查 exit code | ACTIVE | G1' 曾被靜默跳過；新增 --help 守門與 set -e |
| [027](DR-027.md) | 批次腳本的失敗語意；字串替換必須斷言錨點 | ACTIVE | 偽裝成成功的失敗比 crash 危險；憲法 §2.7、§3.5 |
| [028](DR-028.md) | 報告腳本必須有最小資料煙霧測試 | ACTIVE | --help 擋不住需真實資料才觸發的錯；憲法 §3.6 |
| [029](DR-029.md) | 階層架構下 task-IL 主張成立 | ACTIVE | 取代 DR-015 的適用範圍；須同時陳述差距擴大的雙重來源 |
| [030](DR-030.md) | 回報內容必須存在於 committed 產物中 | ACTIVE | 憲法 §2.8；臨時算出的數字必須先寫進產物再回報 |
| [031](DR-031.md) | E1 記憶體曲線以階層版完整重跑（方案 A），排在最後 | ACTIVE | 曲線不可假設可移植；每個 \|M\| 都要報結構性指標 |
| [032](DR-032.md) | 長 job 必須有存活訊號；批次不得用變數當命令 | ACTIVE | zsh 不做 word splitting；outputs/_status/<job>.json |
| [033](DR-033.md) | B1 的論文定位：KD 與 replay 保存不同的東西 | ACTIVE | KD 保選取行為、replay 保任務歸屬；B1 是最不穩的一臂 |
| [034](DR-034.md) | 報告腳本的 fixture 必須涵蓋所有遍歷維度 | ACTIVE | 跨 tag 蒐集要過濾所有語意欄位；狀態檔要主動讀 |
| [035](DR-035.md) | group-level distillation 首次有效驗證 | ACTIVE | task-IL +3.95、class-IL +3.71 皆 5/5；效果不在 Jaccard 上 |
| [036](DR-036.md) | L_sem 在 class-IL 上無可測效果 | SUPERSEDED-BY DR-038 | 措辭含循環論證，由 DR-038 承載修正 |
| [037](DR-037.md) | 階層的價值在於放大方法優勢 | ACTIVE | flat +0.74/+0.75 → hier +3.28；DR-021 的論據補強 |
| [038](DR-038.md) | L_sem 的措辭修正與範圍限定 | ACTIVE | 「在階層架構下」移除不損害準確率；刪去循環論證那句 |

| [039](DR-039.md) | 架構完整性三實驗（G3/G4/G5）的 pre-registration | ACTIVE | q_tau / 狀態迴圈 / group L_sem 全部未經測試；判準先於結果寫定 |

| [040](DR-040.md) | G4 必須自己接上 q_tau；run_exp2 一律餵零向量 | ACTIVE | zeros(512) + 「關閉=填零」⇒ 位元相同 20/20，否則 G4 是保證的 null |

| [041](DR-041.md) | 接線缺口為第四例；§2.9 升格為必須 | ACTIVE | 四例皆非 PI 發現；機制生效性實測改為實驗啟動前的門檻 |

| [042](DR-042.md) | 效率主張改建在 task-IL，倍數 8× → **4×** | ACTIVE | 跨容量配對只有 A5@128−A3@512 為 5/5；A3 曲線未飽和，4× 是下界 |

| [043](DR-043.md) | G345 落判：三元件依判準移出主方法 | ACTIVE | 三者皆 FAIL；但 q_tau 洩漏率 −5.92、group L_sem 配額 KL −0.005 皆 5/5 必須報告 |

| [044](DR-044.md) | Jaccard 隨機參照統一為逐 slide 口徑 | ACTIVE | 舊口徑用 task 平均 n，與觀測值不同口徑；結論方向未變 |

| [045](DR-045.md) | G1 退化率 88.6% 與 84.5% 都對，差在範圍 | ACTIVE | 88.6% = 全部 arm（3708/4185）；84.5% = 只算 A5（1179/1395）；引用須寫明口徑 |

| [046](DR-046.md) | CL 消融 Phase 0/A：測試對照組照准；ΔUtility 取代比值 | ACTIVE | 自建測試對照組允許、改 production 須先問；utility 會變號故比值不可讀 |

**append-only 的範圍**：已寫的卡不改內文；**補記早先的決策是允許的**。
DR 編號不得有缺口，由 `tests/test_ledger.py` 強制。

墓園（柴火區）見 [GRAVEYARD.md](GRAVEYARD.md)；未解釋的現象見 [SEEDS.md](SEEDS.md)。

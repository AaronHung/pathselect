# PROJECT NARRATIVE — 我們怎麼走到今天

> 這不是流水帳，是**決策與證據的因果鏈**。資料來源：`docs/ledger/`（DR-001..042、
> GRAVEYARD、SEEDS、CLAIMS）、`docs/CONSTITUTION.md`、`outputs/` 各報告、git log。
> 撰寫基準：commit 52593fc（2026-08-25）；同日稍晚 `pipeline_g345` 完成
> （4/4 done，見 `outputs/_status/pipeline_g345.json`），G345 結果已併入本檔。
> 只陳述 repo 內實際存在的東西；每個數字附來源檔。

---

## 1. 起點與轉向

前身是 navipath v9（技術報告的 "Ours"）。v9 的程式與被指導教授否決的方法
（QPMIL、Zero*/Router 命名體系）糾纏在一起，無法原地清乾淨，但 v9 的實驗結果
必須留作對照基準。裁定（DR-002，2026-08-20，commit 14acdb3）：

- 開新 repo `pathselect`，方法程式**全部重寫**；
- v9 的 6 個結果檔搬進 `reference/` 並以 `reference/SHA256SUMS.txt` 凍結，唯讀；
- 用紅線測試（`tests/test_no_banned_deps.py`）擋住舊方法識別字回流 ——
  禁令由測試強制，不靠 code review（DR-004，commit 043fba0）。
  Zero*/Router 全面禁用；"Navigation" 保留，但只能用在真正有獨立 coarse view 時
  （GRAVEYARD G-07）。

更早的 DR-001（v0.5）曾把 ICASSP 投稿與老師的任務序列分成 Track A/B 雙軌，
實際結果是兩邊都無法 freeze；DR-003 以 Sol 合併案收斂為單一主線 v0.8，
四位階敘事 specialist → joint → SeqFT → CL，程式 freeze 9/2、論文 8/31。
（DR-001/003 的原始規劃文件在 repo 外的 L4 層，卡上標 TODO。）

搬過來的：v9 的 6 個結果檔與 skill bank（`reference/v9/`）、CONCH 推論程式
（vendored 到 `third_party/`）、資料載入層（`data/`）。留在舊 repo 的：
QPMIL 相依、MoE/機制探測（navipath ADR-0006 已砍，G-02）、多尺度特徵管線（G-03）。
`scripts/v9_reference.py` 是全 repo 唯一允許出現舊方法名的程式檔（唯讀翻譯存檔 key）。

---

## 2. 決策因果鏈（核心）

### (a) 操作點：B=64/c=8 → B=8/c=1（D1 → DR-005）

原契約 B=64/c=8 是在沒有 budget 曲線的情況下訂的。Exp 0 補完
K ∈ {1,2,4,8,16,32,64} 後，四 task 平均在 **K=8 達到峰值 0.8797**、K=16 起下降
（`outputs/exp0/BASELINES.md` 四 task 平均表）→ 裁定 B=8、c=1（DR-005，
commit 58a81ea）：B=64 會讓方法操作在最佳點之外，該區間邊際 patch 是負貢獻；
c=1 讓每輪加入的 patch 佔總證據 1/8，e_t 槓桿最大；B=8 時 c=8 即 one-shot，
statefulness ablation 直接是 c∈{8,4,2,1}。→ 後續所有實驗（S2、Exp2、E1、G 系列）
都建立在這個操作點上。同時期 D2（`outputs/exp0/EFFECTIVE_K.md`）顯示 softmax
權重的 eff_K/K 掉到 0.375，支持 DR-006「softmax 主線依訓練一致性選定、
非依數值選定（softmax 0.8738 < 等權 0.8878 仍照選）」。

### (b) 架構：flat → hier（DR-021 → G1 失敗 → DR-025 → G1' → DR-029/037）

1. Exp 2 七臂全部是 flat（L3b），但定稿架構圖中央是 Group→Patch 階層、標題含
   "Hierarchical" —— 圖文脫節是致命不一致。裁定以階層重跑 CL 主線，判準
   pre-register：「階層 ≥ flat 或在雜訊內皆採用階層」（DR-021，2026-08-24）。
2. **G1 失敗**：hier-A5 − flat-A5 class-IL = −18.69 pp、0/5
   （`outputs/exp2/hier/HIER.md:127`），命中停止分支。但結構性診斷顯示測到的
   不是階層：per_chunk 配額在 c=1 時 largest-remainder 只有一個名額、必然給
   argmax(r)，r 又逐輪不變 → 退化成「先挑一組再取該組 top-8」——
   84.5% 的 slide 只用一組（DR-025 context；hier2 報告中複核為 88.6%，
   `docs/CLAIMS.md` C-02）。
3. 裁定配額口徑改為**對整個 budget**（per_budget，DR-025），DR-021 判準原文
   一字不改沿用 ——「這是修復壞掉的實驗，不是重釣結果」。
4. **G1' 通過結構性把關**：單組比例 2.4%、平均用到 4.31 組
   （`outputs/exp2/hier2/HIER2.md` 結構性診斷節）。判準落點：
   hier-A5 − flat-A5 class-IL = −1.20 pp、1/5（within noise）→ 依 pre-registered
   判準採用階層為主線（DR-029）。
5. 論據後來升級（DR-037）：flat 下 A5−A3 的 task-IL 跨順序一致但微小
   （reverse +0.74 / main +0.75），階層放大到 +3.28（5/5）——
   「階層的價值在於放大方法優勢」，優於 DR-021 原文的「可解釋配額」。
   ⚠️ 同時必須陳述：差距擴大一部分來自 hier-A3 退化（−3.11 pp），
   不全是 A5 變好（DR-029、HIER2.md 結論節）。

### (c) 記憶體效率主張：2× → 8× → 4×（DR-017 → DR-019 → DR-031 → DR-042）

1. 預期最強攻擊是「replay 做了全部的事，把 |M| 開大就好」→ E1 記憶體曲線
   {A3,A5} × |M|∈{64..1024} × 5 seeds 為主要防禦（DR-017）。
2. flat 版判讀（DR-019）：不採 A5@128 vs A3@512 的 4×（踩在 A3@512 低點上，
   reviewer 看得穿），改以 A3 全域最佳（A3@256=0.8203）為基準宣稱 **2×**
   （`outputs/exp2/memory/MEMORY.md:124-128`）；四條可宣稱 + 中段弱證據須揭露。
   同時 F2 發現 A3 在 256→512 **systematic 下滑**（+4.25 pp、5/5）→ 三級 win
   count 規則誕生（DR-020），成因排除 replay 強度混淆後列為 open question
   （SEEDS S-01）。
3. 主線改階層後（DR-029），flat 曲線不可假設可移植（hier-A3 − flat-A3 =
   −3.11 pp 已證明 replay 行為不同）→ E1 以階層版完整重跑 40 輪、排在最後
   （DR-031：會改變方法的實驗先跑，這條只重建防禦論述）。
4. 階層版跑完，報告沿 flat 寫法產生了「A5@128 達 A3 class-IL 全域最佳 → 8×」。
   **8× 被撤回**（DR-042）：錨點 |M|=128 的 class-IL 配對只有 4/5 且
   std(7.71)>mean(7.72)；且階層版 A3 的 class-IL 到 1024 仍在上升
   （256→0.7596、1024→0.7858），「對手已飽和」防禦失效。效率主張改建在
   task-IL，倍數只宣稱到跨容量**配對**支持的程度：唯一 5/5 的是
   A5@128 − A3@512 = +3.06 ± 2.80 pp → **4×，且是下界**
   （`outputs/exp2/memory_hier/MEMORY_HIER.md` 跨容量配對節）。
5. 修訂 A（2026-08-25）：**主從倒轉** —— 主要主張是同容量比較
   （5 個容量 A5 皆優，4 個 systematic，含 1024 的 +2.49），4× 降為輔助主張，
   引用需三個限定（`MEMORY_HIER.md` 記憶體主張節；
   `tests/test_memory_hier_report.py` 以 8 項 mutation 守住「倍數必須來自
   5/5 systematic」）。

### (d) task-IL 主張：不得宣稱 → 解禁（DR-015 → DR-029）

flat 下 A5−A3 task-IL = +0.74 ± 1.93 pp、3/5（`outputs/exp2/main/EXP2.md`
paired 表）→ DR-015 定調「task-IL 不宣稱勝出；replay 回復準確率、KD+eq 回復
選取行為」，並明寫「replay 是很強的 baseline，贏不了它是重要資訊，不藏」。
階層版同一對照 +3.28 ± 2.40、5/5（`HIER2.md`）→ DR-029 把 DR-015 限定於 flat
（改標 SUPERSEDED-BY，內文不動），階層版兩軸皆可宣稱（task-IL +3.28 /
class-IL +5.76，均 5/5），但必須同時陳述差距擴大的雙重來源（見 (b)5）。

### (e) group-KD：從「無效」到「有效」（G0 → DR-022 → DR-035）

1. G0 mutation 實測：flat 模式下把整個 F_g 換成擾動網路，選取**位元相同**
   → F_g 對 flat 選取零影響，L_KD 的 group 項對所有已回報指標影響恰為零
   （`tests/test_arch_switch.py`；`tests/README.md` §1b）。
2. DR-022：既有結果有效不重跑，但 group-KD **從未被測試過** —— 增設 A5nG 臂
   （`l_kd(..., group_weight=0)`，完全不計算）隔離之。在退化階層（G1）下
   首測結果是「未顯示效果」。
3. G1' 通過結構性把關後重測（G1'-b）：hier-A5 − hier-A5nG 於 task-IL
   **+3.95 ± 2.50（5/5）**、class-IL **+3.71 ± 2.66（5/5）**，皆 systematic
   → DR-022 結論作廢（SUPERSEDED-BY DR-035）。
   ⚠️ 效果不顯示在 Jaccard（+0.02、2/5）—— group-KD 保組織層配額分佈、
   patch-KD 保具體 patch 身份，兩層分工不同，這是架構圖 Panel I 兩層設計的
   直接證據（`HIER2.md` 兩層蒸餾的分工節）。

### 其他支線（一句話級）

- **遺忘的證明鏈**：DR-011（specialist ≠ oracle、遺忘只由 SeqFT 證明）→
  S2 三軸全崩（class-IL A1 forgetting esca +71.11 / rcc +59.65 / brca +33.69 pp、
  Jaccard ≈ 0 甚至低於隨機參照、ΣU 變負，`outputs/exp2/seqft/SEQFT.md`）→
  DR-012 三軸並報 + task-IL/class-IL/洩漏率並報（S3 重算，
  `outputs/exp2/seqft/TASK_IL.md`：洩漏率 esca 0.889、rcc 0.636）。
- **q_τ 的降格**：Exp1 Gate 1 未通過（L4 − L3 = −6.02 pp，commit 2ea13f9；
  `outputs/exp1/stage1/RESULTS.md`）→ S1 probe：跨器官 task 在 slide 平均
  特徵上 98.21%、在 group prototype 上 98.57% 線性可分
  （`outputs/exp1/diag/TASK_SEPARABILITY.md`）→ q_τ 在此 benchmark 結構性冗餘，
  主張 PARK 進 G-05，定義本身凍結不動（DR-008）。
- **B1 的定位**：B1 補到 5 seeds 後，KD 保住 Jaccard（0.0725 ≈ A3 的 0.0669）
  但洩漏率 0.3231 是 A5 的 3.2 倍；seed4 的 rcc 有 59/76 張被判成 lung
  → 「KD 與 replay 保存不同的東西」升格為機制對照（DR-033，
  `outputs/exp2/ablation/B1_LANDING.md`）。
- **eq（B2/H 系列）**：A5 − A4 在 reverse/main 方向相反（唯一乾淨的翻轉，
  `outputs/exp2/ORDER_DEPENDENCE.md`）；H1 判準（B2 ≥ A5 才跑 H2）在補足
  5 seeds 後不成立（B2 − A5 class-IL −1.49 pp、3/5，
  `docs/CONSTITUTION.md` §1.3 第二實例）→ H2 未跑。
- **架構完整性收尾（DR-039..041）**：逐格盤點架構圖後，三個從未生效的元件
  進入實測（G5 state / G4 q_tau / G3 group L_sem），判準 pre-register 且判準
  讀法在任何結果產出前修訂為「G5 單軸、G4/G3 雙軸」（DR-039 修訂 A）。
  G4 於啟動前發現接線缺口並修正（DR-040，見 §4）。2026-08-25 晚間三個實驗
  全部落判 **FAIL**（`outputs/exp2/arch/ARCH_COMPLETENESS.md`）：
  G5 兩軸皆 within noise → 移除 Panel E 與 "stateful" 用字；
  G4 僅 class-IL 單軸 4/5（+5.85 pp）、task-IL 2/5 → 依雙軸判準不計為通過，
  q_tau 標為 optional 或移除，寫成有機制解釋的 null（S1 98.2/98.6%）；
  G3 僅 class-IL 單軸 4/5（+1.16 pp）、task-IL 2/5 → 不計為通過，
  主方法維持 patch-only。附帶發現：G4 的洩漏率 −5.92 pp 為 5/5 systematic、
  G3 的配額 KL −0.005 為 5/5，但兩者皆非主要指標，不改變落判。

---

## 3. 被推翻與被取代的決策

status=SUPERSEDED-BY 的卡共 5 張。原卡內文一律不改（append-only，規則 1），
推翻由新卡承載 —— 這一節的重點是：**每次改主意都有寫下依據**。

| 原卡 | 原本裁定 | 被什麼證據推翻 | 新裁定 |
|---|---|---|---|
| DR-001 雙軌並行 | ICASSP 與老師任務序列分軌推進 | 兩邊都無法 freeze（執行事實） | DR-003：單一主線 v0.8，freeze 9/2 |
| DR-015 task-IL 不宣稱 | flat 下 A5−A3 = +0.74 pp（3/5）在雜訊內，只宣稱 class-IL/行為軸 | 階層版同一對照 +3.28 ± 2.40（5/5 systematic，`HIER2.md`） | DR-029：DR-015 限定於 flat；階層版兩軸解禁，但須報差距擴大的雙重來源 |
| DR-019 記憶體效率 2× | A5@128 vs A3 全域最佳(256) → 2×；「A3 在 256 後不再改善」為防禦 | 階層版重驗：①「A3 不再改善」不成立（class-IL 到 1024 仍升）；跨容量配對只有 A5@128−A3@512 是 5/5（`MEMORY_HIER.md` DR-019 重驗節） | DR-042：撤回 8×、效率主張改建 task-IL、倍數 4×（下界）；修訂 A 再把同容量主張升為主 |
| DR-022 group-KD 未顯示效果 | 退化階層（per_chunk、84.5% 單組）下增臂隔離，測得無效 | 通過結構性把關的階層（單組 2.4%）重測：+3.95/+3.71 皆 5/5（`HIER2.md`） | DR-035：group-KD 有效；效果不在 Jaccard，兩層蒸餾分工不同 |
| DR-036 L_sem 無可測效果 | 「HistoSelect 的貢獻在於分組結構而非語意先驗」 | 查證發現該句是**循環論證**（正是在分組壓過 patch 分數的架構裡測 patch 層先驗），且「prior 與架構正交」是未查 code 的臆測 | DR-038：措辭改為「**在階層架構下**，語意先驗的移除不損害準確率」；其餘裁定（弱正則、不得宣稱改善準確率、須報 max_sim 洩漏率最低 9.74）不變 |

此外兩次**主動撤回**（不是卡片被取代，是回報內容被更正）：

1. 「A2 在 main 有效、reverse 無效」——共同子集（seeds 0–2）假象；全 5 seeds
   下 A2−A1 在 reverse 是 −3.09 pp（3/5），逐 seed 幅度橫跨 43 pp。該錯誤已
   進入 PI 的 H-SERIES 指示，撤回並衍生憲法 §1.3、§2.4（DR-024）。
2. 「8× 記憶體效率」——見 §2(c)；撤回寫在 DR-042 與
   `MEMORY_HIER.md` 輔助主張節開頭（「本節取代先前基於 class-IL 的 8× 宣稱，
   該宣稱已撤回」）。

---

## 4. 四個「看起來在運作但沒有」的元件（CLAIMS C-26）

同一家族四例，分屬**架構 / 環境 / 實作 / 接線**四層失效模式。
共同點：規格寫了、架構圖畫了；**四次都不是 PI 發現的**（DR-041）。

### 4.1 group-level KD —— 架構層（無作用空間）

- **怎麼發現**：G0（2026-08-24）。要判斷 flat 下 F_g 有沒有影響，不是讀
  `run_rounds` 分支下結論，而是把整個 F_g 換成「權重放大 5 倍再加噪」的網路。
- **怎麼證實**：替換後選取**位元相同**；反向對照（同樣替換在 hier 下會改變選取）
  排除「替換本身無效」（`tests/test_arch_switch.py`；`tests/README.md` §1b）。
- **怎麼處理**：DR-022 增 A5nG 臂隔離 + EXP2.md 加誠實聲明「已回報的 KD 效果
  僅來自 patch-level」；階層修好後重測，**變成有效元件**（DR-035）。

### 4.2 q_τ task conditioning —— 環境層（資訊冗餘）

- **怎麼發現**：Exp1 Gate 1 失敗（L4 − L3 = −6.02 pp）後不硬拗，改問「task
  identity 是不是本來就免費可得」。
- **怎麼證實**：S1 probe —— 4-way task id 用 slide 平均特徵線性可分 98.21%、
  用 group prototype 98.57%（`outputs/exp1/diag/TASK_SEPARABILITY.md`）。
  q_τ 有實作、有用到，但提供的是模型早已擁有的資訊。
- **怎麼處理**：主張 PARK → G-05（復活條件：同器官多任務 benchmark，S-02）；
  定義凍結不動（DR-008）；G4 在階層下補測（判準：兩軸皆 4/5 才算通過，跑中）。

### 4.3 group-level semantic prior —— 實作層（根本沒寫）

- **怎麼發現**：DR-038 查證「prior 與架構正交」時逐行讀 `l_sem()`，發現原始
  規格的兩項 KL（group + patch）只實作了 patch 項 —— 沒有 r 參數、沒有第二個
  KL（`selector/train.py:78-88`）。
- **怎麼證實**：mutation —— 把 group prototype 擾動 5 倍，L_sem 數值**位元不變**
  （0.0226687789 → 0.0226687789）；反向對照擾動 patch 特徵會變
  （→ 0.020234），證明擾動本身有效（`outputs/exp2/prior/PRIOR.md` 末節）。
- **怎麼處理**：承認 G2 測到的是「半邊的 L_sem」；補寫兩層版
  `selector/sem_loss.py`（beta_g=0 與既有位元相同，由測試守住）作為 G3 的
  消融維度 —— **主方法維持 patch-only，主表不因 G3 改動**（DR-039）。

### 4.4 q_τ 接線 —— 接線層（寫了、開關也會讀，但注入點餵零向量）

- **怎麼發現**：準備 G4 時查 `run_exp2.Ctx`，發現 `q0 = torch.zeros(512)`
  （`scripts/run_exp2.py:120`），而 `use_query=False` 的實作是**填零而不縮短維度**
  （`selector/model.py:44`）—— 只打開開關而不接 `TaskQueryBank`，selector 的
  輸入與關閉時完全一樣。
- **怎麼證實**：`check_state_noop.py` 實測 20 組 —— q_tau=zeros 時與
  use_query=False 選取相同 **20/20**（位元相同）；真 task query 時 16/20
  （4 組不同）（`outputs/exp2/arch/noop_check.json`）。
- **怎麼處理**：G4 由 `wire_task_queries` 以 runtime injection 在四個注入點接上
  真 q_τ（含 continual_terms 用 **entry.tau** 的 query —— 舊樣本屬於自己的
  task），啟動時實跑 leakage 閘門（DR-040）。**既有結果不受影響**：run_exp1
  用真 query，run_exp2 從未開啟 use_query，零向量從未進入任何已發表數字，
  沒有結果需要撤回。但仍計為第四例，因為失效本身是真的（DR-041）。

**共同教訓**（C-26 原文）：架構圖與規格書不是實作的證據。前三層看 code 或設定
可能發現，**第四層只有把開/關兩條路實際跑出來比對才會現形**。這推動 §2.9
「機制生效性實測是實驗啟動的前置門檻」由建議升格為必須（DR-041）。

---

## 5. 方法學紀律的演進 —— 每一條都是被事件逼出來的

`docs/CONSTITUTION.md` 逐條的起因（一句話）：

- **§1.1 三級 win count**：F2 出現 4/5 的未定義區間 + A3 非單調需要定性
  （DR-020）。
- **§1.2 n<5 警語自動加註**：main order／元件消融／E3 只有 3 seeds，3/3 被標成
  systematic 會高估信心（DR-023）。
- **§1.3 跨批次配對以全 seeds 為準**：A2−A1「順序效應」被共同子集 seeds 0–2
  誤導（撤回案）；兩週內同一陷阱第二次出現在 H1 的 B2−A5（DR-024；§1.3 證據鉤子）。
- **§1.4 不報 p 值**：n=5 下顯著性檢定誤導；未配對 mean±std 曾掩蓋 A4−A3 的
  一致小差（−1.04±0.15、0/5）（DR-016）。
- **§1.5 esca 一張 = 6.67 pp**：esca 只有 15 張 test slide 的資料事實
  （常數寫死在 `scripts/run_exp1.py:55`、`run_exp2.py:108`；
  未見對應 DR 卡 —— 不確定其首次裁定出處）。
- **§2.1 逐 slide 存檔**：v9 只存彙總 accuracy，flip 分析只能給區間（DR-012 裁定 C）。
- **§2.2 / §2.3 mutation check（主張與斷言）**：G0 的 F_g 替換實驗證明「讀 code
  推論」不夠；一次靜默失敗的字串替換靠自身疏失才發現（`tests/README.md`；
  SEEDS S-18；commit 1c503c8）。
- **§2.4 撤回程序**：A2−A1 錯誤結論已進入 PI 指示，「撤回不算失分、隱瞞才算」
  （DR-024）。
- **§2.5 隔離前先檢查退化**：G1 的「只開階層、不開 state」方法上正確，但
  per-chunk + c=1 + r 不變三者相乘讓階層根本沒有作用空間（84.5% 單組）（DR-025）。
- **§2.6 宣稱前先驗證非 no-op**：use_state=False 時 c=1 八輪與 c=8 一輪位元相同
  20/20 —— 所有已跑實驗的 chunked loop 是資訊上的 no-op（DR-025；CLAIMS C-01）。
- **§2.7 字串替換必須斷言錨點**：兩週內兩次替換未命中卻靜默通過；最嚴重的一次
  是 run_exp2 的模組層 import 錨點根本不存在，671 條測試全綠、排隊 job 撞上才爆
  （DR-027）。
- **§2.8 回報必須存在於 committed 產物**：G1 的結構性診斷表（84.5% 單組）在回報
  中貼出、PI 據此裁定，但因 replace 錨點失效**從未寫進 HIER.md**（DR-030）。
- **§2.9 生效性實測是實驗啟動門檻**：C-26 第四例（q_τ 接線）——若照原樣跑 G4
  會是構造性 null，算力已花、且事後難再誠實區分（DR-041；此條號為 DR-041 裁定
  當下新增，卡上有註記）。
- **§3.1 forgetting 只算前 T−1 個 task**：最後 task 兩時點相同，A1 恆 0 會稀釋
  量級（寫在 `run_exp2.py:449-451` 與 EXP2.md 欄位口徑；未見對應 DR 卡 ——
  不確定首次裁定出處）。
- **§3.2 負面結果照實報**：DR-015（贏不了 replay 不藏）與 DR-019（中段弱證據
  須揭露）的共同要求。
- **§3.3 參照臂不是 baseline**：Sol 紅隊裁定 specialist ≠ oracle（esca 只有
  120 張訓練資料，replay 臂實質可及跨任務資料）（DR-011）。
- **§3.4 執行期檔案凍結**：編輯 run_exp2.py 時排隊 job 抓到改到一半的版本
  （NameError），G1' 整批被靜默跳過；更危險的是「語法正確但語意不同」的情境
  （DR-026）。
- **§3.5 批次腳本失敗語意**：g1prime.sh 沒有 set -e，crash 後照印「=== G1' 完成
  ===」——偽裝成成功的失敗比 crash 危險（DR-027）。
- **§3.6 報告腳本煙霧測試**：report_hier2 的 `NameError: diag` 要真實資料才觸發，
  `--help` 擋不住；同類錯誤第三次（DR-028）。
- **§3.6b fixture 涵蓋所有遍歷維度**：`KeyError: 'final_task_il'` 只在「兩個
  order 且 seed 數不齊」時可達，單 order fixture 全綠；同日 report_prior 跨 tag
  蒐集漏過濾 allocation，把 G1 退化紀錄當主線臂，產出 −17.5 pp 的假結論
  （DR-034）。
- **§3.7 存活訊號 + 禁用變數當命令**：四段式 pipeline 因 zsh 不做 word
  splitting 在第一行就死，log 無任何標記，PI 手動看 log 才發現（DR-032）；
  附帶條款「狀態檔要主動讀」來自狀態檔正確標了 failed 卻 6 小時無人讀（DR-034 附帶）。

---

## 6. 現在手上有什麼

### 已定案

| # | 主結果 | 支撐實驗 | seeds | 關鍵數字 / win count | DR | 證據檔 |
|---|---|---|---|---|---|---|
| 1 | catastrophic forgetting 存在且三軸皆崩（C-20） | S2 SeqFT，雙 order | 3 | class-IL A1 forgetting esca +71.11 / rcc +59.65 / brca +33.69 pp；Jaccard ≈ 0（esca/rcc 低於隨機參照）；ΣU 由正轉負 | DR-011、DR-012 | `outputs/exp2/seqft/SEQFT.md`、`TASK_IL.md` |
| 2 | 洩漏 100% 可歸因選取漂移（C-22；架構直接後果） | frozen head 構造 + S3 | — | 洩漏率 esca 0.889、rcc 0.636（SeqFT 後） | DR-012 | `outputs/exp2/seqft/TASK_IL.md` |
| 3 | replay 大幅回復準確率、跨順序穩定（C-21） | Exp2 主表 + 順序依賴 | 5（reverse）/5（main A3/A5） | A3−A1 class-IL +36.90（reverse）/ +34.28（main）pp | DR-024 | `outputs/exp2/ORDER_DEPENDENCE.md` |
| 4 | 階層為主線；A5 對 A3 兩軸皆勝（C-10） | G1'（hier2） | 5 | task-IL +3.28 ± 2.40（5/5）、class-IL +5.76 ± 3.42（5/5）；⚠️ 須同報 hier-A3 − flat-A3 = −3.11 pp 與 A5 seed std 縮小（±1.41→±0.77） | DR-021、DR-025、DR-029、DR-037 | `outputs/exp2/hier2/HIER2.md` |
| 5 | group-KD 有效，且與 patch-KD 分工不同（C-03） | G1'-b（A5 vs A5nG，hier） | 5 | task-IL +3.95 ± 2.50（5/5）、class-IL +3.71 ± 2.66（5/5）；Jaccard +0.02（2/5）→ 保配額不保 patch 身份 | DR-022、DR-035 | `outputs/exp2/hier2/HIER2.md` |
| 6 | 記憶體主張（主）：所有 \|M\|∈{64..1024} 下 A5 的 task-IL 皆優於 A3 | E1 階層版 | 5 | 5 容量中 4 個 5/5（64:+4.90、128:+3.73、512:+3.28、1024:+2.49；256:+2.54 為 4/5）| DR-031、DR-042 修訂 A | `outputs/exp2/memory_hier/MEMORY_HIER.md` |
| 7 | 記憶體主張（輔）：4× 記憶體效率（下界、限 task-IL） | E1 階層版跨容量配對 | 5 | A5@128 − A3@512 = +3.06 ± 2.80（5/5）；8× 不成立（4/5）；A3 曲線未飽和 | DR-042 | 同上 |
| 8 | KD 與 replay 保存不同的東西（C-24） | B1 落點分析 | 5 | B1 Jaccard 0.0725 ≈ A3 0.0669；B1 洩漏率 0.3231 = A5 的 3.2 倍；⚠️ B1 為最不穩臂（class-IL std ±0.1042） | DR-033 | `outputs/exp2/ablation/B1_LANDING.md` |
| 9 | L_sem 為弱正則：階層下移除不損害準確率（C-25 改寫） | G2 三臂 prior（hier） | 5 | discriminative − none class-IL = −0.02（3/5）；task-IL disc − max_sim = +0.76（5/5，量級極小）；⚠️ 限階層、且測的是「半邊的 L_sem」；須報 max_sim 洩漏率最低 9.74 | DR-007、DR-036、DR-038 | `outputs/exp2/prior/PRIOR.md` |
| 10 | G5：state 條件化無可測增益 → 移除 "stateful" 用字 | G345（hier_state） | 5 | task-IL −0.50（2/5）、class-IL +1.05（3/5），皆 within noise → pre-registered FAIL | DR-039 | `outputs/exp2/arch/ARCH_COMPLETENESS.md` |
| 10b | G4：q_τ 條件化不計為通過（僅 class-IL 單軸有效，照實報） | G345（hier_query，真 q_τ 已接線） | 5 | task-IL +0.31（2/5）、class-IL +5.85（4/5）→ 雙軸判準 FAIL；洩漏率 −5.92 pp（5/5）為次要指標 | DR-039、DR-040 | `outputs/exp2/arch/ARCH_COMPLETENESS.md` |
| 10c | G3：group 層 L_sem 不提供增益 → 主方法維持 patch-only | G345（A5g，beta_g=0.1） | 5 | task-IL −0.52（2/5）、class-IL +1.16（4/5）→ 雙軸判準 FAIL；配額 KL −0.005（5/5）為次要指標 | DR-039 | `outputs/exp2/arch/ARCH_COMPLETENESS.md` |
| 11 | learned 選取遠勝 random；K 曲線峰值 K=8 | Exp 0 | random 5 seeds | K=8：learned 0.8797 vs random 0.6607（+21.90 pp）；峰值非飽和 | DR-005 | `outputs/exp0/BASELINES.md` |
| 12 | q_τ 在本 benchmark 結構性冗餘 | Gate 1 + S1 probe | 3 | L4 − L3 = −6.02 pp；task id 線性可分 98.21%（slide 平均）/ 98.57%（prototype） | DR-008、DR-010 | `outputs/exp1/stage1/RESULTS.md`、`outputs/exp1/diag/TASK_SEPARABILITY.md` |
| 13 | beta_u=0.1 全臂保留；S2 降級 preliminary；A1↔S2 位元一致 | verify_a1 | 3 | 1707 筆全欄位零差異（beta_u=0 下） | DR-014 | `scripts/verify_a1_matches_s2.py`、`outputs/exp2/main/EXP2.md` |

### 進行中

| # | 項目 | 狀態 | 判準 | 出處 |
|---|---|---|---|---|
| 1 | G5 的效果量重測（訓練後模型、真實 slide） | ARCH_COMPLETENESS.md 承諾「G5 跑完後補進本節」；未見產物 —— 不確定是否已排程 | no-op 檢查的 4/20 是下界不是效果量（PI 裁定 2） | `outputs/exp2/arch/ARCH_COMPLETENESS.md` G5 前置節 |
| 2 | G345 產物與最終報告尚未 commit（含 `A5g_*`、`A5_*_hier_query` seed3/4、更新後的 ARCH_COMPLETENESS.md） | git status 未入庫；憲法 §2.8 要求回報內容存在於 committed 產物 | — | `git status` |
| 3 | 論文寫作：完整版母本 → venue 視圖 | freeze 9/2、論文 8/31（今日 8/25） | 不為 venue 砍 idea | DR-003、DR-018 |

（G4、G3 已於 2026-08-25 落判，移入上方「已定案」10b/10c。）

### 已放棄 / 已撤回

| # | 項目 | 為什麼 | 出處 |
|---|---|---|---|
| 1 | 8× 記憶體效率 | 錨點 4/5 且 std>mean；「對手已飽和」防禦在階層失效 | DR-042 |
| 2 | 2× 效率（flat 版四條可宣稱） | 限定於 flat；階層版逐條重驗後 ① 不成立、② 改裁 4× | DR-019（SUPERSEDED）、DR-042 ④ |
| 3 | sequential acquisition 敘事（C-01） | use_state=False 時 chunked loop 位元等同 one-shot（20/20）；G5 又落 FAIL → 替代說法「budgeted top-K selection under a shared frozen head」 | DR-025、DR-039；`docs/CLAIMS.md` C-01 |
| 4 | 「A2 有順序效應」 | 共同子集假象；全 5 seeds 為 −3.09 pp（3/5），幅度橫跨 43 pp | DR-024 |
| 5 | eq 的跨順序穩定性 | A5 − A4 在兩軸皆翻轉（唯一乾淨翻轉）；flat 證據 | `outputs/exp2/ORDER_DEPENDENCE.md` |
| 6 | H2（B2 記憶體曲線） | H1 判準不過：B2 − A5 class-IL = −1.49 pp（3/5，5 seeds） | `scripts/decide_h2.py`；憲法 §1.3 |
| 7 | task-conditioning 有效之主張 | S1 可分離性 → PARK G-05；復活條件為同器官 benchmark | DR-008、G-05 |
| 8 | "stateful" / Panel E | G5 pre-registered FAIL | `outputs/exp2/arch/ARCH_COMPLETENESS.md` |
| 9 | L_sem 改善準確率之宣稱 | G2 全部 within noise（class-IL） | DR-036/038；CLAIMS C-25 |
| 10 | "task-conditioned" 用字（架構圖 q_τ） | G4 雙軸判準 FAIL（task-IL 2/5）；q_τ 標為 optional 或移除，寫成有機制解釋的 null，同器官設定列 future work | `outputs/exp2/arch/ARCH_COMPLETENESS.md` |
| 11 | 完整兩層 L_sem 進主方法 | G3 雙軸判準 FAIL；報告載明機制解釋：classification 下 q_τ 任務內為常數，cos(g_j, q_τ) 退化為 8 組靜態排序 | `outputs/exp2/arch/ARCH_COMPLETENESS.md` |

---

## 7. 沒有走的路

### GRAVEYARD（決定不做；被砍不等於被否證，每條有復活條件）

`docs/ledger/GRAVEYARD.md` 共 11 條。可分三類：

- **前朝遺產，永久出局**：G-01 QPMIL（老師否決，殘留由 DR-002/003 清除）、
  G-02 MoE experts（navipath 時代已砍）。
- **成本被叫停，留規格**：G-03 MLLM-HWSI 多尺度管線（抽取成本；若復活可直接解
  G-07）、G-04 PathAgent 完整實作（前處理更重；G-12 種子留 scaffold 位）。
- **被證據降格，鉤子最肥**：G-05 **q_τ 主張**（S1 98.2/98.6% 線性可分；復活
  條件 = 同器官多任務標籤 BRCA subtype/grade/receptor —— 卡上明寫「最肥的一根
  柴」）、G-06 FiLM 救 q_τ（同因）、G-07 "Navigation" 用字（g_j 由同批 patch
  算出，nothing hidden；復活條件 = 獨立 coarse 表徵）、G-08 per-task LoRA bank
  當方法（需 task id、儲存線性成長 → 永久降級為 specialist reference）、
  G-09 L5/L6 進正文（8/22 調度讓位給 SeqFT 主軸 → 事後看，G5 的 FAIL 證明
  這個讓位沒有損失正文結果）、G-10 mllm can_dataset 當 cohort（每 task test
  僅 6 張）、G-11 牌二多樣性取樣（未被否證，只是排序在牌一之後）。

### SEEDS（跑出來但沒解釋的現象；「還不知道」而非「決定不做」）

`docs/ledger/SEEDS.md`：S-01 有完整證據鉤子 —— **replay buffer 容量與 task
組成的交互作用**：A3 的 class-IL 在 256→512 systematic 下滑（+4.25 ± 2.27、
5/5），已排除 replay 強度混淆，且對照組 A5 沿同區間平穩 —— 現象只發生在純
replay 臂（來源 DR-017/020；資料 `outputs/exp2/memory/MEMORY.md`）。

苗圃 S-02..S-18 一行式登錄，擇要：S-02 同器官多任務 CL suite（= G-05 復活條件，
CVPR 主實驗候選）；S-03 frozen-evaluator 洩漏率當通用 CL 指標；S-04 記憶體
效率前沿當評估軸；S-05 utility-gated KD（牌一，已實作備援未打）；S-07 KD 的
準確率–行為 trade-off（A4−A3：task-IL 0/5 −1.04 但 Jaccard 5/5 +0.07）；
S-08 LoRA sequential-merge 動力學（A2≈A1）；S-16 stateful 的 per-round 監督
（last-round-only 只覆蓋 1/8 決策點 —— G5 FAIL 後這是 L6 復活的先決修正）；
S-18 mutation-checked 斷言實務（本專案 68 條 ledger 測試 + 「數值不可分辨的
違例」陷阱，可寫成 research engineering 短文）。

維護規則：被本篇採用 = 從 SEEDS 移除；被證據砍掉 = 移入 GRAVEYARD 附復活條件。

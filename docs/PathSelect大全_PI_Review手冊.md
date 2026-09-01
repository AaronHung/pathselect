# PathSelect 大全 — 代號、機制、結果、還缺什麼（PI Review 手冊）
**Fable｜2026-08-31｜每個數字皆對 repo（tag `dr046-phaseA`）與 dossier 核實**

---

## §0 三十秒故事線（整篇論文的主張鏈，人話）

> 看病理切片要先「挑證據」——幾千塊小圖裡選 8 塊。我們把其他所有東西凍住（編碼器、診斷規則全不訓練），**唯一會學的就是「怎麼挑」**，所以任何成績變化都能歸因到「看哪裡」。任務一個接一個來的時候：①不保護就慘掉（0.48）；②只換 LoRA+合併這套裝備、不開保護，一樣慘（0.45）——**合併是底盤不是煞車**；③真正的煞車是三件事：複習舊樣本＋模仿自己當年的挑法＋不准效用退步（0.82，遺忘砍半，而且「還挑不挑得到當年那些證據」這件事我們直接量了——保護臂高出無保護臂兩個數量級）。④老師問的每個「為什麼不那樣做」我們都變成事先寫死規則的實驗：先打地基？測了，沒差，留現行。一顆 LoRA 用到底？測了，輸 3.5 分，換新有理。拼裝題翻盤且更有趣：沒保護時拼裝大勝裸序列（序列會自我覆寫），但全配置仍決定性勝拼裝——序列的價值在於能搭載保護。折減題：維持全額合併。

---

## §1 臂大全（「臂」= 一種訓練配方；每臂 5 seeds 除非另註）

**讀法**：B 結尾＝bare 裸版（關掉所有保護損失）。「HOW」欄＝老師往下鑽時你的答案。

### 1a. 參照臂（不是 baseline——DR-011，這是老師可能挖的坑）

| 代號 | 論文名 | 人話 | 為什麼跑 | HOW | 結果與狀態 |
|---|---|---|---|---|---|
| **R1** | Per-task specialist | 每個 task 各養一個專屬模型 | 參照點 | 每 task 用**自己那份**資料（120–774 張）獨立全參數訓練，互不相干 | class-IL 0.8777（全場最高，這是它的參考意義）。**⚠️ 它不是 task-IL 上界**：R1 只看得到自己 task 的小 cohort，而 A3/A5 透過 replay 實質接觸跨任務資料——所以 A5 task-IL 0.9147 > R1 0.9027 **不是異常**，老師問就這樣答 |
| **R2** | Joint (offline) | 四個 task 資料全部混在一起學 | 「如果沒有先後順序」的參照 | 一次看全部 2273 張、8 類一起訓 | 0.7789。**不可用來論遺忘**（它沒有順序，只證明多任務互相干擾的存在） |

### 1b. 主階梯（CL 主表的骨幹，flat 底盤——見 §4「flat/hier」條）

| 代號 | 論文名 | 人話 | 為什麼跑 | HOW | 結果 |
|---|---|---|---|---|---|
| **A1** | Sequential fine-tuning (SeqFT) | 裸奔：按順序全參數硬訓，零保護 | 證明遺忘真的發生（地板） | 兩顆 MLP 全參數，task 換了就直接繼續訓 | class-IL 0.4774、遺忘 0.5193、Jaccard 0.0023（幾乎完全換掉當年選的證據） |
| **A2** | + LoRA merge | 換上 LoRA+合併裝備，但不開任何保護 | **關鍵對照**：證明合併本身不防遺忘 | 每 task 只訓 rank-4 小殘差（16,400 參數＝2.1%），task 結束把殘差**恆等寫回**底座（W←W+BA，函數不變），換新殘差 | 0.4466，與 A1 **無系統性差異**（−3.09pp、3/5、seed 間橫跨 43pp——所以嚴禁說誰好誰壞，DR-024 教訓）。遺忘 0.5798 一樣高 → 「**merge 是 substrate 不是機制**」 |
| **A3** | + Replay | A2＋複習舊樣本 | 複習值多少 | 每個訓練步從記憶庫抽 1 筆舊 slide，把**一般診斷損失**也算在它身上（同一次反向傳播，沒有多吃梯度步） | 0.7778，大回血；遺忘降到 0.1102 |
| **A4** | + Selection distillation | A3＋模仿自己當年的挑法 | 蒸餾再加多少 | 對重播的舊 slide，把**當年存的分數快照**（組分數 r、patch 分數 s）當老師，KL 拉齊現在的分數——**雙層**：組層＋patch 層；teacher 不是另一顆網路，就是快照 | 0.7972；Jaccard 0.1752（全場最高——最會「照舊挑」） |
| **A5** | **Ours (full)** | 全配置＝A4＋效用保護 | 主角 | 再加一個**單向 hinge**：只懲罰「效用比當年退步」，進步不罰 | **0.8239**、遺忘 **0.0539**（A3 的一半）、洩漏 0.1005、task-IL 0.9147。A5−A1 = **+34.64pp、5/5** |

### 1c. 機制拆解臂（回答「三個保護件各自幹嘛、缺誰不行」）

| 代號 | 論文名 | 人話 | 為什麼跑 | HOW | 結果（給老師的機制故事） |
|---|---|---|---|---|---|
| **B1** | KD-only | 只留蒸餾（複習損失、效用保護都關） | 拆出蒸餾單獨值多少 | 記憶庫照常填、舊樣本照常取（蒸餾需要它），但只算 KD 損失 | **最漂亮的機制證據**：Jaccard 0.0725 ≈ A3——選取行為保住了；但洩漏 0.3231 = A5 的 3.2 倍、class-IL 只 0.5999。落點分析（DR-033）：某 seed 的 RCC 76 張裡 **59 張被推去 LUNG 的類別**——不是亂猜（亂猜會散開）。→ **蒸餾保住「怎麼挑」，保不住「證據屬於哪個任務」；補任務歸屬的是 replay**。兩者不可互換 |
| **B2** | Utility-only | 只留效用保護 | 拆出效用項 | 只算 hinge | 0.8089、洩漏 0.0906（很低）。**統計紀律的救命故事**：3 seeds 時 B2 還贏 A5，補到 5 seeds 翻盤（A5−B2 +1.49、2/5）——差一點就用 3 顆 seed 把方法砍掉一個組件 |
| **A5nG** | Ours − group-KD | A5 但蒸餾只留 patch 層、關掉組層 | 證明**雙層**蒸餾必要 | KD 的 group 項係數設 0（位元等價於不算），patch 項不變 | 關掉組層蒸餾損 task-IL 3.95／class-IL 3.71（**皆 5/5**，階層配置）→ 兩個決策層（給誰配額／組內挑誰）都需要保護 |

### 1d. DR-046 新臂（老師 8/28 質詢直接對應的實驗；W/L 已完成，C/A5H 今晚）

| 代號 | 論文名 | 人話 | 回答老師哪句 | HOW | 結果／狀態 |
|---|---|---|---|---|---|
| **W1** | Warm-start variant | **老師的直覺**：task1 先全參數把 MLP 練起來當地基，task2 起才 LoRA | 「第一個 task 一定要 train MLP 吧？」 | stage 0 全參數訓練→掛上 LoRA（掛上瞬間函數不變，有自檢）→之後照 A5 | class-IL 差 −1.80（W1 只贏 2/5）→ **無系統性差異，依事前寫死的分裂規則留用現行**，還多一個「每 task 只訓 2.1%」賣點。次要發現：Jaccard 升到 0.245（4/5）、ΔU 全場最佳——**改變挑什麼、沒改變結果** |
| W1B | （消融內） | W1 的裸版 | 拆「地基 vs 保護」 | 同 W1 但保護全關 | 比 A2 好 +2.86（4/5）、洩漏 −1.59（4/5）→ **地基只在沒保護時有用；「preservation subsumes initialization」** |
| **L2** | Single-adapter variant | 一顆 LoRA 從頭用到尾、不換新、最後才合併 | 「為什麼每個 task 要換新 LoRA 再 merge？」 | 跳過每個界線的合併與重置，同一組 A,B 連訓四個 task | 輸 A5：class-IL −3.54（4/5）、task-IL −2.06（4/5）→ **換新有理**。機制解釋：不換新＝總漂移永遠鎖在 rank 4；每 task 換新＝可累積到 rank 16——**持續學挑證據需要隨任務成長的容量** |
| L2B | （消融內） | L2 的裸版 | 同上拆解 | 同 L2、保護全關 | 略優於 A2（task-IL +1.37、4/5）——沒保護時「不重置」有一點連續性好處，但有保護時容量勝出 |
| **C1** | Post-hoc composition (sum) | 四顆 LoRA **互不認識**、各自從同一起點只學自己的 task，最後把四個增量**相加**拼成一個模型 | 「merge 是 sequential 還是一次？」的第二義 | 各 task 獨立訓練（保護損失**天生開不了**——沒有「舊模型」可蒸餾）→ 增量求和。跟同樣裸的 A2 比才公平；容量上限同為 rank 16，所以比的純粹是「**按順序學**」值多少 | **翻盤**：C1 四指標全 5/5 勝 A2（class-IL +24.6、task-IL +6.6、洩漏 −24.9、Jaccard +0.27）→ 依翻盤條款誠實報告。但 A5 仍決定性勝 C1（class-IL +13.2、洩漏 −12.6 皆 5/5）——精煉主張：「序列的價值來自能搭載保護；保護只在序列中可定義」。C1 Jaccard 0.275 全場最高、成績中段（穩≠準第二例） |
| **C2** | Post-hoc composition (mean) | 同 C1 但取**平均** | 「不用 average？」第一義 | 重用 C1 的增量快取、零訓練 | 與 C1 無系統性差異——sum/mean 可互換 |
| **A5H** | Damped merging (α=0.5) | 照常訓練，但每次合併只把殘差**寫進一半** | 「不用 average／折減？」第二義 | merge 係數 α 從 1 改 0.5（一個參數） | class-IL 噪音內（2/5）；α=1 於 task-IL +0.98、洩漏 −1.15 皆 4/5 → **維持 α=1**。Jaccard 升 0.238（穩≠準第三例） |
| **L2@hier**（選配，已核准） | — | 把 L2 搬到**階層底盤**重跑 | 保險：確認「要換新 LoRA」不是扁平底盤特有 | `--arch hier`，檔名帶 `_hier` 後綴、零覆蓋 | 排在 Phase B 之後，約 2.5 小時 |

### 1e. 補充實驗族（不在主表但撐柱子的）

| 名稱 | 人話 | 關鍵數字 | 撐哪根柱子 |
|---|---|---|---|
| **Exp0 / B-curve** | 「學出來的挑法」贏過「隨機挑」多少、預算 B 掃描 | 28/28 格全勝；B=8 到峰值 0.8797 | 挑證據是可學的；B=8 的選擇依據 |
| **S1 probe** | 任務身分從特徵本身就猜得出來嗎 | 線性探針：slide 平均 98.21%、組原型 98.57% | 解釋 q_τ 為何沒作用空間（見 G4） |
| **HIER2** | 階層 vs 扁平的正式對照 | A5−A3 從 +0.74(3/5) 放大到 task-IL +3.28／class-IL +5.76（皆 5/5）；絕對值差在噪音內、階層洩漏略高 +2.15 | 階層的採用理由（放大並穩定方法優勢） |
| **MEMORY_HIER** | 記憶庫容量掃描（64→1024） | 5 個容量 A5 全勝 A3（4 個 5/5）；**A5@128 > A3@512（+3.06 task-IL，5/5）** | 「機制設計可勝過把 buffer 開大」 |
| **ORDER_DEPENDENCE** | 換任務順序還成立嗎 | A3−A1 跨順序穩如山（+36.9/+34.3）；A5−A3 class-IL「reverse 有效、main 無效」（+4.61/−0.28） | 誠實極限：效用項的增量對順序敏感（與 A5−A4 翻轉同源） |
| **E3** | β_u（效用項權重）消融，3 seeds | β_u=0.1 vs 0：task-IL +1.12（3/3），其餘噪音 | L_util 定位為輔助項，不用力宣稱 |
| **DR-047**（選配未跑） | replay 抽幾筆、記憶庫換不換 policy | — | 「比例多少」的敏感度備援 |

---

## §2 Gate 大全（「gate」= 事前寫死的採用/淘汰判準；先寫規則、再看數據、照規則執行）

| Gate | 測什麼 | 判準（事前寫死） | 結果 | 現在住哪 |
|---|---|---|---|---|
| **G1** | 階層第一版（per-chunk 配額） | 階層 ≥ flat 才採用 | **FAIL**（−18.69、0/5）——但診斷發現測到的不是階層：配額機制退化成 84.5% 的 slide 只用一組 | 教訓入憲法：「隔離變因前先確認機制沒退化」；設計錯誤記 DR-025 |
| **G1'** | 階層第二版（per-budget 配額，判準原封沿用） | 同上 | **PASS**：單組比例降到 2.4%、平均用 4.31 組；絕對值與 flat 打平但方法對比放大至 5/5 | 階層成為方法定型；主表仍在 flat（見 §4） |
| **G3** | group 層語意先驗（組層 L_sem） | 雙軸（task-IL＋class-IL）皆需受益 | **FAIL**（task-IL −0.52）；但留下 5/5 次要發現：配額分佈 KL **降低** 0.005 | 移出主方法；次要發現進 Removed Components |
| **G4** | q_τ 任務條件查詢向量 | 同上雙軸 | **FAIL**；次要發現：洩漏 −5.92pp（5/5）；S1 解釋：任務身分本來就 98% 可解碼 | 移出；輸入零填充；圖上不畫（⛔清單第 1 條） |
| **G5** | 狀態化（逐輪重算）選取 | 同上 | **FAIL** | 移出；主線為一次性 largest-remainder 配額 |
| **G-W1** | task-1 warm-start | class-IL ≥4/5 才換主法；分裂→留現任 | **分裂（2/5）→ 留現任** | 消融節；task-1 protocol 明寫於 Method |
| **G-L2** | 單顆 adapter vs 每 task 換新 | A5 ≥4/5 → 換新正當化；L2 ≥4/5 → 改採更簡設計 | **A5 勝 4/5 → 換新正當化（容量論證）** | 消融節；L2@hier 保險排程中 |
| **G-C1** | 拼裝 vs 按順序 | A2 ≥4/5 → 預註冊句；C1 勝 → 翻盤條款 | **翻盤觸發**：C1 全 5/5 勝 A2 → 誠實報告＋主張精煉；A5 仍 5/5 勝 C1 | 消融節（改寫版） |
| **G-α** | 折減合併 α=0.5 | A5(α=1) ≥4/5 → 維持；否則 PI 決策 | class-IL 分裂；兩副軸 4/5 挺 α=1 → 建議維持（待 PI 一字核可） | 消融節 |

---

## §3 術語辭典（人話＋HOW＋追問應答）

| 術語 | 人話 | HOW／追問應答 |
|---|---|---|
| **flat / hier（底盤）** | 挑 patch 的兩種方式：flat＝全場混著排名取前 8；hier＝先給各組織區配名額、再各區內挑 | 主表的 35 個跑歷史上做在 flat；階層依 G1' 被採用為方法定型，理由是**放大並穩定**方法對比（不是絕對值更高——絕對值在噪音內、洩漏還略高，兩面都報）。老師問「消融為何在 flat」→「消融延伸的是 §4.4 那張正典主表，同底盤才公平；階層保險（L2@hier）另跑」 |
| **LoRA** | 不動大權重，只學一個 rank-4 的小殘差（兩個小矩陣 B·A） | 每個 Linear 層各掛一組；每 task 新參數 16,400＝選擇器的 2.1%；α=r 所以尺度=1 |
| **merge（合併）** | task 結束把殘差寫回底座：W ← W + BA | **恆等改寫**——寫回瞬間函數一個位元都不變（訓練時本來就是算 W+BA）。寫到哪：兩顆 MLP 的**每一層 Linear 權重** |
| **substrate（底盤）** | 「merge 是底盤不是煞車」 | 證據：A2 與 A1 無系統性差異、遺忘一樣高。merge 買到的是**單一模型、task-free、常數成本推論**；煞車是三個保護損失 |
| **bare（裸）／B 後綴** | 同款臂但保護全關 | 用來把「裝備」與「保護」的貢獻拆開 |
| **warm-start** | task1 先全參數打地基 | 老師的直覺；W1 測完：無系統性差異、留現行 |
| **post-hoc composition（拼裝）** | 各學各的、最後拼 | 與 sequential 差的不只時機：訓練基底、舊知識存在與否、KD 對象、優化路徑全都不同——所以**嚴禁**叫它「merge timing」 |
| **等價命題** | 「留著舊 adapter 最後一次合併」＝「逐界合併」 | 同一個總和的兩種記法（帳本比喻）→ 前向逐位相同 → 梯度相同 → 整條軌跡相同。unit test 6/6，**省下一整支實驗**。老師「一次 merge?」拆兩義：這義＝數學等價；拼裝義＝C1 實驗 |
| **behavioral snapshot（行為快照）** | 記憶庫存的不是圖，是「當年怎麼挑」的證據：組分數 r、patch 分數 s、效用 u、選了哪些 index | 特徵憑 slide_id 重載，所以 512 筆的庫極輕；蒸餾的 teacher 就是快照，**不存在第二顆網路** |
| **reservoir（水塘抽樣）** | 記憶庫滿了怎麼汰換：任何時刻都是「至今所有經驗」的均勻樣本 | 各 task 佔比∝貢獻量（對經驗均勻、不對 task 均勻——早期小 cohort 會被稀釋，明講） |
| **replay_k=1** | 每個訓練步抽 1 筆舊的 | 新舊 1:1；三個保護損失跟當前樣本**同一次反向傳播**——replay 臂沒有多吃梯度步，比較公平 |
| **leakage（洩漏）** | 預測落到「別的任務的類別」的比例 | 收窄後的正確說法：**同資料同標籤空間下、跨方法的差異**可歸因選取；沿時間的變化混有「標籤空間變大」效應——所以才需要 Jaccard/ΔU 這種繞過分類頭的直接量測 |
| **q_τ / e_t（已移除件）** | 任務查詢向量／選取狀態 | 皆 gate FAIL、輸入零填充。q_τ 留下 −5.92pp 洩漏的 5/5 次要發現；解釋（非證明）：任務身分本來 98% 可解碼 |
| **DR / ledger / graveyard** | 決策卡帳本（append-only）／淘汰件墳場 | 每個設計決策有卡、有判準、有出處；被淘汰的組件連同其數據**保留展示**——removal ≠ hiding |
| **pre-registration（預註冊）** | 先寫死規則再跑實驗 | 對老師的最強一句：「規則是看到數據**之前**寫的，哪邊贏用哪邊」 |

---

## §4 指標大全

| 指標 | 人話 | 怎麼算 | 為什麼要它 |
|---|---|---|---|
| class-IL（=A_Final） | 期末考：所有見過的類一起考 | 最終階段、聯集標籤空間的準確率 | CL 的主判準 |
| task-IL | 開卷小考：只在自己 task 的兩類內選 | 各 task 自己類對內的準確率 | 任務內鑑別力有沒有壞 |
| leakage | 答案填到別人的考卷去 | 預測落在錯誤任務類別集的比例 | 「證據的任務歸屬」壞沒壞（B1 的故事） |
| Forgetting | 曾經會、後來忘了多少 | 每個舊 task：歷史最高分 − 期末分，取平均（完整準確度矩陣本來就在 log 裡） | CL 標準指標 |
| Plasticity | 新東西還學得動嗎 | 剛學完當下的分（矩陣對角線）平均 | 證明保護不是靠犧牲新任務 |
| **Selection Jaccard** | **還挑不挑得到當年那些證據**（本文獨有） | 同一張 slide：剛學完時選的 8 塊 vs 期末選的 8 塊，交集/聯集 | 「忘記去哪裡看」的直接量測；無保護 0.001–0.002 vs 保護 0.16 |
| **ΔUtility** | 挑出來的證據還有沒有當年好用 | 期末效用 − 當年效用（負＝退步） | 全臂皆負（誠實），但保護臂少一個數量級（A5 −16.8 vs A2 −300.8） |

---

## §5 統計紀律（老師問「為什麼不報 p 值」的完整答案）

1. **5 seeds、配對比較、win-count 三級制**：5/5＝systematic（可用力宣稱）；4/5＝directional（方向性，措辭降級）；≤3/5＝不宣稱。n=5 下跑 t-test 是假精確，所以**描述優先、不報 p 值**——這是寫死的政策不是偷懶。
2. **一張 slide 的粒度**：ESCA 測試集只有 15 張，一張＝6.67pp——比這小的差異連一張切片都翻不動，不解讀。
3. **兩個救命故事**（紀律的實地效果）：**DR-024**——A2−A1 曾被 3 顆共同 seed 誤判成「順序效應」，補齊後撤回（逐 seed 橫跨 43pp）；**B2 教訓**——3 seeds 時 B2 還贏 A5，5 seeds 翻盤，差點錯砍一個組件。老師若挑 seeds 少，就把這兩個故事講給他聽：**正因為 seed 少，我們才用這麼嚴的宣稱紀律**。

---

## §6 版圖：做了什麼✅／進行中🔄／還缺什麼⬜

**✅ 已完成且可用的彈藥（review 時的火力清單）**
- 主階梯全套（A1–A5＋R1/R2）＋新四欄（Forgetting/Plasticity/Jaccard/ΔU 全部零 GPU 離線重算）
- 機制拆解（B1 的「保挑法不保歸屬」落點分析、B2、A5nG 雙層蒸餾 5/5）
- 階層採用鏈（G1 敗→診斷→G1' 過，兩面誠實）
- 記憶體效率（A5@128>A3@512）、順序依賴（含誠實極限）
- W1（老師直覺，測完：留現行＋兩個次要發現）、L2（換新 LoRA 有理，4/5）
- 等價命題 unit test、S1 probe、Exp0 B-curve
- 論文：spconf 版 5 頁已編譯、Sol 紅隊 4 個 P0 已修、venue 事實已官方查證（ICASSP 2027、9/16、4+1、單盲、spconf）

**✅ Phase B 完成**（C1/C2/A5H 落地、paper/ 已入庫、EXP2 全臂重產）；**🔄** L2@hier（~17:40）

**⬜ 凍結前（9/2 目標）**：C/α gate 判定入 ledger → **全稿去黑話＋新 Abstract**（我做）→ verify 腳本掃 main.tex（數字對 repo、\pending 清零）→ 你的總 review ＋ 老師毛病清單校準
**⬜ 投稿前**：圖 v1.0（照修正清單）、壓縮 10–15%（刀口：圖 caption／Related Work／§3.4）、2027 kit 換裝（上線監看中）、作者名單、bib 欄位終核
**⬜ 未來／擴充版**：DR-047、EWC 族、任務順序全排列、extended_master.tex 復活成 arXiv 版

---

## §7 去黑話對照表（論文用語規範——內部代號僅准活在 repo 與擴充版附錄）

| 內部 | 論文正文用語 |
|---|---|
| A1 | sequential fine-tuning (SeqFT) |
| A2 | LoRA-and-merge（或 the merge-only substrate） |
| A3 / A4 | + replay ／ + selection distillation |
| A5 | our full method（Ours） |
| A5nG | without group-level distillation |
| B1 / B2 | distillation-only ／ utility-only |
| W1 / W1B | the warm-start variant（／without preservation） |
| L2 / L2B | the single-adapter variant（／without preservation） |
| C1 / C2 | post-hoc composition by summation ／ averaging |
| A5H | damped merging (α=0.5) |
| 5/5、4/5 | "in all five seeds" ／ "in four of five seeds"（Abstract 一律不出現） |
| reverse order | the primary task sequence |

---

## §8 新 Abstract 草稿（頂刊語域；逐句結構標註；你點頭我就換上）

> **[動機]** Whole-slide images are diagnosed under a budget: a few evidence patches are selected, and everything downstream depends on them. **[缺口]** When diagnostic tasks arrive sequentially, existing continual learning protects classifiers or representations, leaving a distinct failure invisible---the model may simply stop looking in the right place. **[洞見/設計]** We expose this failure by construction: the encoders and the diagnostic rule are frozen, so the only learnable behavior is a hierarchical selection policy, and any cross-method difference in performance is attributable to where the model looks. **[方法-載體]** Each task trains a small low-rank residual that is folded exactly into one shared selector, giving task-free, constant-cost inference. **[轉折]** Merging alone, however, provides no systematic protection. **[方法-機制]** Preservation comes from replaying stored behavioral snapshots---scores and indices, never images---through selection distillation and a utility-preserving objective. **[結果]** On four TCGA tasks, this raises final class-incremental accuracy by 34.6 points over sequential fine-tuning in every seed, halves forgetting relative to replay alone, and retains two orders of magnitude more of the originally selected evidence.

（約 150 字、零代號、零黑話；「in every seed」承載 5/5 而不露統計行話；「provides no systematic protection」合紅線。）

---

## §9 老師最可能追問的 12 題速答卡

1. **你們 task-IL 怎麼比 per-task specialist 還高？**——R1 只看得到自己 task 的 120–774 張；我們經 replay 實質接觸跨任務資料。R1 的參考意義在 class-IL（0.8777 全場最高），這點我們在文中明寫（DR-011）。
2. **merge 到底改了哪些權重？**——兩顆 MLP 的每一層 Linear；寫回是恆等改寫，函數不變，附 unit test。
3. **task1 不先訓 MLP？**——測了（W1）：無系統性差異（2/5），依事前規則留現行；地基只在沒保護時有用（W1B 對 A2 +2.86、4/5）。
4. **為何每 task 換新 LoRA？**——測了（L2）：不換新輸 3.5 分（4/5）；機制＝總漂移 rank 4 vs 16 的容量。
5. **memory 存什麼、比例多少？**——不存圖不存特徵，存行為快照；每步抽 1 筆、同一次反向傳播；全域水塘、各 task 佔比∝貢獻。
6. **推論用不用 memory／task-ID？**——都不用：單一合併選擇器、常數成本；memory 只活在訓練期。
7. **為什麼不報 p 值？**——5 seeds 下 t-test 是假精確；改用配對 win-count 三級制＋兩個救命故事（DR-024、B2）。
8. **效用項到底有沒有用？（A5−A4 翻轉）**——誠實答：class-IL 增量對任務順序敏感（reverse 有效、main 無效），文中如實報；跨順序穩固的是 replay（+34–37）與整體保護包。
9. **主表為何 flat、方法卻是 hier？**——主表是歷史正典（單調階梯在此建立）；階層依預註冊判準採用，理由是放大並穩定方法對比（+3.28/+5.76 皆 5/5），絕對值在噪音內且洩漏略高——兩面都報；消融同底盤延伸主表，另有 L2@hier 保險。
10. **leakage 能全歸因 selection 嗎？**——跨方法、同標籤空間：能；沿時間：混有標籤空間擴張效應——所以我們補了 Jaccard/ΔU 直接量選取。
11. **任務身分 98% 可解碼，那 q_τ？**——正因如此它沒有作用空間（gate FAIL、移除），但留下 −5.92pp 洩漏的 5/5 次要發現，附錄保留。
12. **只有 CONCH、只有 B=8，普適嗎？**——明列為 limitation；B=8 有 Exp0 峰值依據；本 benchmark 任務可分性偏高，更難的混合任務可能重開任務條件化的空間——文中原句。

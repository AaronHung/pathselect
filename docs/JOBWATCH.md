# 從新終端機看長跑 job（SOP）

長跑佇列是用 `nohup` 起的，**跟終端機完全脫鉤**：關掉 iTerm/WezTerm 視窗、
關掉 Claude Code、甚至那個 session 結束了，job 都照跑。所以「看 job」
不需要接回原本的 session，開一個新的視窗就行。

## 一鍵

```bash
cd ~/research/02_pathselect
bash scripts/jobwatch.sh        # 一次性快照
bash scripts/jobwatch.sh -f     # 快照 + 即時跟著現在這一步（Ctrl-C 離開）
```

`-f` 的 Ctrl-C 只停掉 `tail`，**不會動到 job**。

## 手動版

| 想知道 | 指令 |
|---|---|
| 還活著嗎 | `pgrep -f sota_queue.sh` |
| 跑到第幾步 | `tail -3 logs/sota_queue.log` |
| 這一步在做什麼 | `tail -f logs/sota/<name>.log` |
| 完成幾個產物 | `/bin/ls outputs/exp2/sota/per_slide/ \| wc -l` |
| 有沒有失敗 | `cat logs/sota/FAILED.txt` |
| 防睡眠還在嗎 | `pgrep -x caffeinate` |
| **全部停掉** | `kill -TERM -$(ps -o pgid= -p $(cat logs/sota_queue.pid) \| tr -d ' ')` |

## 四個會咬人的細節

### 1. `kill <pid>` 停不乾淨

行程樹長這樣（實測）：

```
3665  bash scripts/sota_queue.sh          ppid=1（已脫離終端機）
├─ 3667  caffeinate -is bash …            持有防睡眠 assertion
└─ 6187  python scripts/run_exp2.py …     真正在算的，每一步換一個新的
```

三個**同一個 process group**（PGID 3663）。Unix 殺父行程不會連帶殺子行程，
所以 `kill 3665` 之後 python 6187 會變成孤兒**繼續佔著 CPU 跑完這一折**，
而佇列已經不會再往下走了 —— 最糟的狀態。

正確做法是殺整個 group（pid 前面加負號）：

```bash
kill -TERM -3663
```

`jobwatch.sh` 每次都會把當下正確的那行印出來，照抄即可。

### 2. `pgrep -f` 會命中兩個，`pgrep -x` 只命中一個

```console
❯ pgrep -f sota_queue.sh
3665
3667
❯ pgrep -x caffeinate
3667
```

不是 bug，是兩個旗標比對的**欄位不同**：

* **`-f`** = 比對**完整命令列**（full command line）。
  caffeinate 的命令列是 `caffeinate -is bash scripts/sota_queue.sh` ——
  整串裡也含 `sota_queue.sh`，所以一起被命中。
* **`-x`** = **精確**比對**行程名**（`comm`，就是 `ps -o comm=` 那一欄），
  不看參數。3665 的行程名是 `bash`、3667 的是 `caffeinate`，
  所以 `-x caffeinate` 只中一個。

驗證：

```console
❯ ps -o pid,comm,command -p 3665,3667
  PID COMM        COMMAND
 3665 bash        bash scripts/sota_queue.sh
 3667 caffeinate  caffeinate -is bash scripts/sota_queue.sh
```

`pgrep -x bash` 也會中很多不相干的 shell，所以兩個都不能單獨用來鎖定佇列。
`jobwatch.sh` 的做法是 `-f` 先撈、再用 `comm` 過濾出 `bash` 那一個。

### 3. 為什麼不在 repo 根目錄也不會 file not found

因為 **`pgrep` 的參數不是檔名，是 regex pattern**。它從來不去開檔案，
只是拿這個 pattern 去比對「核心裡每個行程的命令列字串」。
所以在哪個目錄下跑都一樣，也不需要那個檔案真的存在。

副作用：pattern 是 regex，`.` 是萬用字元。`sota_queue.sh` 其實會匹配
`sota_queue-sh`、`sota_queueXsh` 之類。要精確就寫 `pgrep -f 'sota_queue\.sh'`。
這裡不會撞到，所以沒有特別加。

對照組：`tail logs/sota_queue.log` **就是**檔名，換個目錄就會 file not found。
所以 SOP 第一行才要 `cd ~/research/02_pathselect`。

### 4. 互動 shell 的 `ls` 是 eza 別名

```console
❯ type ls
ls is an alias for eza --icons=always
```

eza 會在檔名前加圖示字元。塞進 `$( )` 之後 `tail` 會說找不到檔案：

```bash
tail -1 "$(ls -t logs/sota/*.log | head -1)"   # ❌ No such file or directory
tail -1 "$(/bin/ls -t logs/sota/*.log | head -1)"  # ✅
```

**寫腳本一律用 `/bin/ls`**（腳本裡不會載入 `.zshrc` 的別名，但手打指令會，
一旦把手打的內容貼進腳本就會踩到，所以乾脆統一）。

## Activity Monitor 看得到嗎

看得到「有沒有在跑」，看不到「跑到哪一折」。

* 搜尋 **`python`** → 會看到一個 CPU 約 100%、記憶體數 GB 的 `python`。
  這是**目前這一步**的 `run_exp2.py`，每完成一折就換一個新的（pid 會變）。
* 搜尋 **`caffeinate`** → 確認防睡眠還在。
* 搜尋 **`bash`** → 佇列本身，CPU 幾乎 0%（它只負責依序叫 python）。

判斷是否正常：CPU 持續在燒 = 正常；掉到 0% 且 log 長時間不動 = 可能卡住。

⚠️ **不要用 Activity Monitor 的「強制結束」停 job** —— 它一次只殺一個行程，
會踩到上面第 1 點的孤兒問題。用 `kill -TERM -<PGID>`。

⚠️ Activity Monitor 給不了「現在是 fold 4 的 stage 2、loss 0.79」這種資訊。
**判斷跑得順不順一律看 log，不看 Activity Monitor。**

## 闔蓋

`caffeinate -is` 的兩個旗標（man page 原文）：

* `-i` — prevent the system from **idle** sleeping
* `-s` — prevent the system from sleeping，**只在接 AC 電源時有效**

⚠️ **不接電源時 `-s` 是無效的**，只剩 `-i` 擋閒置睡眠。
本專案實際發生過闔蓋後 job 受影響的情形，所以純電池＋闔蓋不要賭。
（clamshell sleep 的確切機制我沒有實測驗證，這裡只陳述已知的事實。）

真的要闔蓋帶著走，先停乾淨再走 —— 佇列**可續跑**（產物已存在的 run 自動跳過），
回來重新啟動不會重跑已完成的部分：

```bash
kill -TERM -<PGID>
# 回來後
nohup caffeinate -is bash scripts/sota_queue.sh > logs/sota_queue.log 2>&1 &
echo $! > logs/sota_queue.pid
```

⚠️ 重啟會**覆寫** `logs/sota_queue.log`（`>` 不是 `>>`），進度計數會從頭算。
產物不受影響。想留舊的先改名。

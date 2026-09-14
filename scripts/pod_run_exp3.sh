#!/usr/bin/env bash
# DR-052 pod 批次執行器（在 pod 的 tmux 視窗內以 nohup 跑）。
#
#   nohup bash scripts/pod_run_exp3.sh <batch_id> [parallel] [threads] > logs/pod/batch<id>.log 2>&1 &
#
# * run 清單由 scripts/pod_exp3_batches.py <batch_id> 產生（每行 name|args）。
# * 每個 run：outputs/exp3/runs/<name>/ 下寫 meta.json（seed、參數、commit、主機、時間）、
#   run.log；成功（exit 0 且 per_slide 檔存在）才 touch DONE。已有 DONE 的 run 跳過。
# * 以 xargs -P 平行跑；每個 run 固定 OMP_NUM_THREADS（訓練路徑是 CPU，見 DR-052）。
# * 批次結束在 logs/pod/heartbeat.log 追加一行 BATCH_DONE <id> <done>/<total>。
#   （pod 無法主動 rsync 到 Mac —— Mac 在 NAT 後；由 Mac 端看到 BATCH_DONE 後 pull。）
set -uo pipefail
cd "$(dirname "$0")/.."
BATCH="${1:?batch id}"; PAR="${2:-10}"; THREADS="${3:-16}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
mkdir -p logs/pod outputs/exp3/runs
LIST="logs/pod/batch${BATCH}.runs"
if [ -n "${EXP3_LIST:-}" ]; then cp "$EXP3_LIST" "$LIST"; else   # EXP3_LIST：煙霧測試用自訂清單
  python scripts/pod_exp3_batches.py "$BATCH" > "$LIST" || { echo "批次清單產生失敗"; exit 1; }
fi
TOTAL=$(wc -l < "$LIST" | tr -d ' ')
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
echo "[$(date '+%F %T')] batch $BATCH：$TOTAL run，parallel=$PAR threads=$THREADS commit=$COMMIT"

run_one() {
  local line="$1"
  local name="${line%%|*}" args="${line#*|}"
  local dir="outputs/exp3/runs/$name"
  mkdir -p "$dir"
  if [ -f "$dir/DONE" ]; then echo "[$(date '+%F %T')] ▷ skip $name（DONE）"; return 0; fi
  local seed; seed=$(echo "$args" | sed -n 's/.*--seeds \([0-9]*\).*/\1/p')
  python - "$dir/meta.json" "$name" "$seed" "$args" "$COMMIT" <<'PY'
import json, sys, socket, datetime
p, name, seed, args, commit = sys.argv[1:]
json.dump({"run": name, "seed": int(seed), "args": args, "commit": commit,
           "host": socket.gethostname(), "started": datetime.datetime.now().isoformat(),
           "head": "accumulating", "device": "cpu",
           "omp_threads": __import__("os").environ.get("OMP_NUM_THREADS")},
          open(p, "w"), indent=1)
PY
  echo "[$(date '+%F %T')] ▶ start $name"
  # shellcheck disable=SC2086
  python scripts/run_exp2.py $args > "$dir/run.log" 2>&1
  local rc=$?
  local tag; tag=$(echo "$args" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')
  local nfiles; nfiles=$(ls "outputs/exp3/$tag/per_slide/" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$rc" -eq 0 ] && grep -q "class-IL=" "$dir/run.log"; then
    date '+%F %T' > "$dir/DONE"
    echo "[$(date '+%F %T')] ✅ done $name（per_slide 檔數 $nfiles）$(grep 'eval' "$dir/run.log" | tail -1)"
  else
    echo "[$(date '+%F %T')] ❌ FAIL $name rc=$rc；最後 5 行："; tail -5 "$dir/run.log" | sed 's/^/    /'
  fi
}
export -f run_one; export COMMIT
xargs -P "$PAR" -I{} bash -c 'run_one "$@"' _ {} < "$LIST"
DONE=$(for n in $(cut -d'|' -f1 "$LIST"); do [ -f "outputs/exp3/runs/$n/DONE" ] && echo x; done | wc -l | tr -d ' ')
echo "[$(date '+%F %T')] BATCH_DONE $BATCH $DONE/$TOTAL" | tee -a logs/pod/heartbeat.log

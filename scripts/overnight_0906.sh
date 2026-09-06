#!/usr/bin/env bash
# DR-048 Prompt 10 C 段：過夜佇列。
#
#   佇列 A  A3 flat forward folds 1–10 → A1 flat forward folds 1–10
#   佇列 B  外部基準的重現檢查（scripts/repro_external.py：clone 到 repo 之外、
#           先 smoke 再決定要不要跑十折；不可行就寫紀錄並正常結束）
#   末步    sota/report_sota.py 重產（**不 commit** —— 明天由 PI 觸發）
#
# 規矩同 scripts/sota_queue.sh：可續跑（產物已存在即跳過，由 run_exp2 內建
# resume 負責）、逐 run 一份 log、單一步驟失敗**不中斷**整批、失敗清單另存。
set -u
cd "$(dirname "$0")/.."

TAG="${SOTA_TAG:-sota}"
LOGDIR="logs/sota"
FAILED="$LOGDIR/FAILED_overnight_0906.txt"
mkdir -p "$LOGDIR"
: > "$FAILED"

step=0
run() {                       # run <名稱> <指令...>
  local name="$1"; shift
  step=$((step + 1))
  printf '[%s] #%02d %s\n' "$(date '+%F %T')" "$step" "$name"
  if ! "$@" > "$LOGDIR/$name.log" 2>&1; then
    printf '    ❌ 失敗 → %s\n' "$LOGDIR/$name.log"
    echo "$name" >> "$FAILED"
  fi
}

echo "═══ 佇列 A-1：A3 flat forward folds 1–10"
for k in 1 2 3 4 5 6 7 8 9 10; do
  run "A3_flat_fwd_f$k" python scripts/run_exp2.py --arms A3 --order main \
      --arch flat --fold "$k" --seeds "$k" --tag "$TAG"
done

echo "═══ 佇列 A-2：A1 flat forward folds 1–10"
for k in 1 2 3 4 5 6 7 8 9 10; do
  run "A1_flat_fwd_f$k" python scripts/run_exp2.py --arms A1 --order main \
      --arch flat --fold "$k" --seeds "$k" --tag "$TAG"
done

echo "═══ 佇列 B：外部基準重現檢查（clone 在 repo 之外）"
run "repro_external" python scripts/repro_external.py

echo "═══ 末步：重產 SOTA 表（不 commit）"
run "report_sota_overnight" python sota/report_sota.py --tag "$TAG" --order reverse

n=$(wc -l < "$FAILED" | tr -d ' ')
printf '[%s] 過夜佇列結束：%d 步，失敗 %s 個\n' "$(date '+%F %T')" "$step" "$n"
[ "$n" -gt 0 ] && cat "$FAILED"
echo "（依裁定不 commit —— 明天由 PI 以「Prompt 11 回報」觸發彙整）"
exit 0

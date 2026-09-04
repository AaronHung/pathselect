#!/usr/bin/env bash
# DR-048 C9：SOTA 主表的實驗佇列。
#
# 協定（PI 補充裁定）：
#   * 順序 = reverse（對應基準論文 Tab. 2）—— 每個 run 都顯式帶 --order reverse
#   * 平均對象 = 10 折，每折一個 run，**seed = 折號**
#   * 訓練設定維持本 repo 的：5 epoch、lr 1e-3、rank 4（run_exp2.py 的預設）
#
# 佇列順序（PI 指定）：
#   1. A5 flat reverse folds 1–10
#   2. A5 hier reverse folds 1–10
#   3. zeroshot（reverse，每折）
#   4. opcm（依 C1 delta 快取 → **DR-046 協定 fold 1 / seed 0–4**，表中標明）
#   5.（選配）A5 flat forward folds 1–10
#   6.（選配）A1 / A3 flat reverse folds 1–10
#
# 可續跑：產物已存在的 run 直接跳過（run_exp2.py 內建 resume；sota/*.py 亦同）。
# 逐 run 一份 log 在 logs/sota/。任何一個 run 失敗**不中斷佇列**，
# 失敗清單寫進 logs/sota/FAILED.txt，最後一併回報。
set -u

cd "$(dirname "$0")/.."
TAG="${SOTA_TAG:-sota}"
LOGDIR="logs/sota"
FAILED="$LOGDIR/FAILED.txt"
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

echo "═══ 1. A5 flat reverse folds 1–10"
for k in 1 2 3 4 5 6 7 8 9 10; do
  run "A5_flat_rev_f$k" python scripts/run_exp2.py --arms A5 --order reverse \
      --arch flat --fold "$k" --seeds "$k" --tag "$TAG"
done

echo "═══ 2. A5 hier reverse folds 1–10"
for k in 1 2 3 4 5 6 7 8 9 10; do
  run "A5_hier_rev_f$k" python scripts/run_exp2.py --arms A5 --order reverse \
      --arch hier --fold "$k" --seeds "$k" --tag "$TAG"
done

echo "═══ 3. zero-shot（reverse，每折；seed = 折號）"
for k in 1 2 3 4 5 6 7 8 9 10; do
  run "zeroshot_rev_f$k" python sota/zeroshot.py --order reverse \
      --fold "$k" --tag "$TAG"
done

echo "═══ 4. OPCM（DR-046 協定：fold 1 / seed 0–4，依 C1 delta 快取）"
run "opcm_rev_dr046" python sota/opcm.py --order reverse --seeds 0,1,2,3,4 \
    --fold 1 --tag "$TAG"

echo "═══ 5.（選配）A5 flat forward folds 1–10 —— 對應基準論文 Tab. 1"
for k in 1 2 3 4 5 6 7 8 9 10; do
  run "A5_flat_fwd_f$k" python scripts/run_exp2.py --arms A5 --order main \
      --arch flat --fold "$k" --seeds "$k" --tag "$TAG"
done

echo "═══ 6.（選配）A1 / A3 flat reverse folds 1–10"
for arm in A1 A3; do
  for k in 1 2 3 4 5 6 7 8 9 10; do
    run "${arm}_flat_rev_f$k" python scripts/run_exp2.py --arms "$arm" \
        --order reverse --arch flat --fold "$k" --seeds "$k" --tag "$TAG"
  done
done

echo "═══ 產表"
run "report_sota" python sota/report_sota.py --tag "$TAG" --order reverse

n=$(wc -l < "$FAILED" | tr -d ' ')
printf '[%s] 佇列結束：%d 步，失敗 %s 個\n' "$(date '+%F %T')" "$step" "$n"
[ "$n" -gt 0 ] && cat "$FAILED"
exit 0

#!/usr/bin/env bash
# DR-056 第 7 步的 run 3／run 4 重啟：第一次啟動誤用 --order forward（本 codebase 的
# 正向順序叫 main，見 run_exp2.py 的 ORDERS），兩個 run 在參數分派就 KeyError，
# 未進入任何運算。與 run 1／run 2 同一台、同一批、同時在跑。
set -u
cd /workspace/pathselect
source /workspace/venvs/ps4/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PATHSELECT_TORCH_THREADS=8
mkdir -p logs/exp4
COMMON="--arms A5 --seeds 1 --fold 1 --arch hier --head accumulating --uold current \
--mem-capacity 128 --out-root outputs/exp4"

launch () {
  local name=$1 order=$2 tag=$3; shift 3
  ( date +%s > logs/exp4/${name}.start
    python scripts/run_exp2.py $COMMON --order "$order" --tag "$tag" "$@" \
      > logs/exp4/${name}.log 2>&1
    echo $? > logs/exp4/${name}.rc
    date +%s > logs/exp4/${name}.end ) &
}
launch run3_rfull_fwd main cand_m128_rfull
launch run4_r256_fwd  main cand_m128_r256 --replay-candidate-only
wait
date +%s > logs/exp4/BATCH_FWD.end
for f in logs/exp4/run3*.rc logs/exp4/run4*.rc; do echo "$f -> $(cat "$f")"; done
echo "FWD DONE"

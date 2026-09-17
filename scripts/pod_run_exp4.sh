#!/usr/bin/env bash
# DR-056 第 7 步：四個 probe run，同批同機、四路平行。
#   run 1  |M|=128 讀整張   reverse  → outputs/exp4/cand_m128_rfull
#   run 2  |M|=128 只讀候選 reverse  → outputs/exp4/cand_m128_r256
#   run 3  |M|=128 讀整張   forward（--order main）→ outputs/exp4/cand_m128_rfull
#   run 4  |M|=128 只讀候選 forward（--order main）→ outputs/exp4/cand_m128_r256
# 共同設定：A5、hier、accumulating、--uold current、B=8、fold 1、seed 1。
set -u
cd /workspace/pathselect
source /workspace/venvs/ps4/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PATHSELECT_TORCH_THREADS=8
mkdir -p logs/exp4

COMMON="--arms A5 --seeds 1 --fold 1 --arch hier --head accumulating --uold current \
--mem-capacity 128 --out-root outputs/exp4"

launch () {                       # launch <name> <order> <tag> [extra flags]
  local name=$1 order=$2 tag=$3; shift 3
  ( date +%s > logs/exp4/${name}.start
    python scripts/run_exp2.py $COMMON --order "$order" --tag "$tag" "$@" \
      > logs/exp4/${name}.log 2>&1
    echo $? > logs/exp4/${name}.rc
    date +%s > logs/exp4/${name}.end ) &
}

launch run1_rfull_rev  reverse cand_m128_rfull
launch run2_r256_rev   reverse cand_m128_r256  --replay-candidate-only
launch run3_rfull_fwd  main    cand_m128_rfull
launch run4_r256_fwd   main    cand_m128_r256  --replay-candidate-only

wait
date +%s > logs/exp4/BATCH.end
for f in logs/exp4/run*.rc; do echo "$f -> $(cat "$f")"; done
echo "BATCH DONE"

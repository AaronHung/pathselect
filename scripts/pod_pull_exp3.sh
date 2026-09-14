#!/usr/bin/env bash
# 把 pod 的 outputs/exp3/ 與 logs/pod/ 拉回本機（Mac 在 NAT 後，只能由 Mac 端 pull）。
#   bash scripts/pod_pull_exp3.sh            # 只拉，不改任何既有檔（rsync 不帶 --delete）
set -euo pipefail
cd "$(dirname "$0")/.."
POD="${POD:-root@157.157.221.30}"; PORT="${PORT:-56428}"; KEY="${KEY:-$HOME/.ssh/id_ed25519}"
RS=(rsync -a --stats -e "ssh -p $PORT -i $KEY")
mkdir -p outputs/exp3 logs/pod
"${RS[@]}" "$POD:/workspace/pathselect/outputs/exp3/" outputs/exp3/ | grep -E "Number of files transferred|Total transferred"
"${RS[@]}" "$POD:/workspace/pathselect/logs/pod/" logs/pod/ | grep -E "Number of files transferred"
echo "DONE 標記：$(find outputs/exp3/runs -name DONE 2>/dev/null | wc -l | tr -d ' ')；per_slide 檔："
for d in outputs/exp3/*/per_slide; do [ -d "$d" ] && echo "  $d: $(ls "$d" | wc -l | tr -d ' ')"; done

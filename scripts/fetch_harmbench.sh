#!/usr/bin/env bash
# Fetch the HarmBench text behaviour sets.
#
# HarmBench: Mazeika et al., 2024 (arXiv:2402.04249), MIT licensed.
# https://github.com/centerforaisafety/HarmBench
set -euo pipefail

DEST="${1:-data}"
BASE="https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets"

mkdir -p "$DEST"
for split in val test; do
  echo "Fetching harmbench_behaviors_text_${split}.csv ..."
  curl -fsSL "${BASE}/harmbench_behaviors_text_${split}.csv" \
    -o "${DEST}/harmbench_behaviors_text_${split}.csv"
done

echo "Done. Files in ${DEST}:"
ls -la "$DEST"/harmbench_behaviors_text_*.csv

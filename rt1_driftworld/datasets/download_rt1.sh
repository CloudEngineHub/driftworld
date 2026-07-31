#!/usr/bin/env bash
set -euo pipefail

RAW_HOME="./rt1/raw_tfds"
SRC="gs://gresearch/robotics/fractal20220817_data/0.1.0"
DST="$RAW_HOME/fractal20220817_data/0.1.0"

mkdir -p "$DST"
echo "Downloading $SRC -> $DST"

if command -v gcloud >/dev/null 2>&1; then
  gcloud storage cp -r -anon "$SRC/*" "$DST/"
elif command -v gsutil >/dev/null 2>&1; then
  gsutil -o "GSUtil:parallel_thread_count=16" \
         -o "GSUtil:parallel_process_count=8" \
         -m cp -n -r "$SRC/*" "$DST/"
else
  echo "ERROR: neither 'gcloud' nor 'gsutil' found." >&2
  exit 1
fi

echo "Done. Raw data under: $DST"

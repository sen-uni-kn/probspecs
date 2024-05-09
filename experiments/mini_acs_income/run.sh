#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

OUT_DIR="$HERE/../output/$TIMESTAMP/mini_acs_income"
mkdir -p "$OUT_DIR"

TIMEOUT=3600
HARD_TIMEOUT=3660s  # 60 seconds for setup, etc (very generous)

for nv in {1..8}
do
  timeout "$HARD_TIMEOUT" python -u "$HERE/verify.py" \
  --num-variables "$nv" \
  --timeout "$TIMEOUT" \
  "$@" \
  | tee "$OUT_DIR/${nv}_variables.log"
done

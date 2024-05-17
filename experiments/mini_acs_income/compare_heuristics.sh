#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

OUT_DIR="$HERE/../output/$TIMESTAMP/mini_acs_income/verify"
mkdir -p "$OUT_DIR"

TIMEOUT=1200
HARD_TIMEOUT=1260s  # 60 seconds for setup, etc (very generous)

for nv in {1..6}
do
  timeout "$HARD_TIMEOUT" python -u "$HERE/verify.py" \
  --num-variables "$nv" \
  --timeout "$TIMEOUT" \
  "$@" \
  | tee "$OUT_DIR/${nv}_variables.log"
done


#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

OUT_DIR="$HERE/../output/$TIMESTAMP/mini_acs_income/verify"
mkdir -p "$OUT_DIR"

TIMEOUT=3600
HARD_TIMEOUT=3660s  # 60 seconds for setup, etc (very generous)

# Networks of different sizes
for size in {1000..10000..1000}
do
  timeout "$HARD_TIMEOUT" python -u "$HERE/verify.py" \
  --num-variables 4 \
  --timeout "$TIMEOUT" \
  --size "${size}" \
  "$@" \
  | tee "$OUT_DIR/4_variables_${size}_neurons.log"
done

# Networks of different depths
for depth in {2..10}
do
  timeout "$HARD_TIMEOUT" python -u "$HERE/verify.py" \
  --num-variables 4 \
  --timeout "$TIMEOUT" \
  --depth "${depth}" \
  "$@" \
  | tee "$OUT_DIR/4_variables_${depth}_layers.log"
done

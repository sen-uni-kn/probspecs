#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

OUT_DIR="$HERE/../output/$TIMESTAMP/acasxu/robustness"
mkdir -p "$OUT_DIR"

TIMEOUT=900
PRECISION=0.001
for label in {0..4}
do
  for target in {0..4}
  do
    for i in {0..4}
    do
      python -u "$HERE/robustness.py" \
      --network "1_1" --label "$label" --target "$target" --input "$i" \
      --timeout "$TIMEOUT" --precision "$PRECISION" \
      "$@" \
      | tee "$OUT_DIR/net1_1_${label}_to_${target}_${i}.log"
    done
  done
done

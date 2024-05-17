#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

TIMEOUT=60
OUT_DIR="$HERE/../output/$TIMESTAMP/acasxu/safety/$TIMEOUT"
mkdir -p "$OUT_DIR"
for i1 in {2..5}
do
  for i2 in {1..9}
  do
    python -u "$HERE/safety.py" \
    --network "${i1}_${i2}" --property 2 --timeout "$TIMEOUT" \
    "$@" \
    | tee "$OUT_DIR/property2_${i1}_${i2}.log"
  done
done

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

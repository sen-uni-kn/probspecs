#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

OUT_DIR="$HERE/../output/$TIMESTAMP/acasxu/safety"
mkdir -p "$OUT_DIR"

TIMEOUT=900
PRECISION=0.01
for i2 in {1..9}
do
  python -u "$HERE/safety.py" \
  --network "2_${i2}" --property 2 --timeout "$TIMEOUT" --precision "$PRECISION" \
  "$@" \
  | tee "$OUT_DIR/property2_2_${i2}.log"
done
python -u "$HERE/safety.py" \
--network "1_9" --property 7 --timeout "$TIMEOUT" --precision "$PRECISION" \
"$@" \
| tee "$OUT_DIR/property7_1_9.log"
python -u "$HERE/safety.py" \
--network "2_9" --property 8 --timeout "$TIMEOUT" --precision "$PRECISION" \
"$@" \
| tee "$OUT_DIR/property8_2_9.log"

OUT_DIR="$HERE/../output/$TIMESTAMP/acasxu/robustness"
mkdir -p "$OUT_DIR"

# Comparison with eProVe
OUT_DIR="$HERE/../output/$TIMESTAMP/acasxu/safety_less_precise"
mkdir -p "$OUT_DIR"

TIMEOUT=45  # median runtime of eProVe
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

#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

# Comparison with ProVe_SLR and eProVe.
# Run prob-specs for a certain time budget
for TIMEOUT in 10 30 60 600 3600
do
  OUT_DIR="$HERE/../output/$TIMESTAMP/acasxu/safety/$TIMEOUT"
  mkdir -p "$OUT_DIR"
  for net in "4_3" "4_9" "5_8"
  do
    python -u "$HERE/safety.py" \
    --network "$net" --property 2 --timeout "$TIMEOUT" \
    "$@" \
    | tee "$OUT_DIR/property2_$net.log"
  done
  python -u "$HERE/safety.py" \
  --network "1_9" --property 7 --timeout "$TIMEOUT" --precision "$PRECISION" \
  "$@" \
  | tee "$OUT_DIR/property7_1_9.log"
  python -u "$HERE/safety.py" \
  --network "2_9" --property 8 --timeout "$TIMEOUT" --precision "$PRECISION" \
  "$@" \
  | tee "$OUT_DIR/property8_2_9.log"
done

# Extended comparison with eProVe
# median runtime of eProVe is 45s
for TIMEOUT in 10 30 60
do
  OUT_DIR="$HERE/../output/$TIMESTAMP/acasxu/safety/$TIMEOUT"
  mkdir -p "$OUT_DIR"
  for i1 in 2 3
  do
    for i2 in {1..9}
    do
      python -u "$HERE/safety.py" \
      --network "${i1}_${i2}" --property 2 --timeout "$TIMEOUT" \
      "$@" \
      | tee "$OUT_DIR/property2_${i1}_${i2}.log"
    done
  done
  # 4_3 and 4_9 already run above
  for i2 in 1 2 4 5 6 7 8
  do
    python -u "$HERE/safety.py" \
    --network "4_${i2}" --property 2 --timeout "$TIMEOUT" \
    "$@" \
    | tee "$OUT_DIR/property2_4_${i2}.log"
  done
  # 5_8 already run above
  for i2 in 1 2 3 4 5 6 7 9
  do
    python -u "$HERE/safety.py" \
    --network "5_${i2}" --property 2 --timeout "$TIMEOUT" \
    "$@" \
    | tee "$OUT_DIR/property2_5_${i2}.log"
  done
done

# Comparison with SpaceScanner
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

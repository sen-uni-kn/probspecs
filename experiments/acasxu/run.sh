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
for net in "2_1" "2_2" "2_3" "2_4" "2_5" "2_6" "2_7" "2_8" "2_9"
do
  python "$HERE/safety.py" -u \
  --network "$net" --property 2 --timeout "$TIMEOUT" --precision "$PRECISION" \
  "$@" \
  | tee "$OUT_DIR/${net}_property2.txt"
done
for net in "1_9" "net_1_9_property_7_partially_repaired_1" "net_1_9_property_7_partially_repaired_2" "net_1_9_property_7_partially_repaired_3"
do
  python "$HERE/safety.py" -u \
  --network "$net" --property 7 --timeout "$TIMEOUT" --precision "$PRECISION" \
  "$@" \
  | tee "$OUT_DIR/${net}_property7.txt"
done
for net in "2_9" "net_2_9_property_8_unknown"
do
  python "$HERE/safety.py" -u \
  --network "$net" --property 8 --timeout "$TIMEOUT" --precision "$PRECISION" \
  "$@" \
  | tee "$OUT_DIR/${net}_property8.txt"
done

OUT_DIR="$HERE/../output/$TIMESTAMP/acasxu/robustness"
mkdir -p "$OUT_DIR"

TIMEOUT=900
PRECISION=0.001
for label in {0..4}
do
  for target in {0..4}
  do
    for i in {0..5}
    do
      python "$HERE/robustness.py" -u \
      --network "1_1" --label "$label" --target "$target" --input "$i" \
      --timeout "$TIMEOUT" --precision "$PRECISION" \
      "$@" \
      | tee "$OUT_DIR/1_1_${label}_to_${target}_${i}.txt"
    done
  done
done

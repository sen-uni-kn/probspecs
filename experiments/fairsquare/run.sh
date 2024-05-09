#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

OUT_DIR="$HERE/../output/$TIMESTAMP/fairsquare"
mkdir -p "$OUT_DIR"

TIMEOUT=900
HARD_TIMEOUT="960s"  # 60 secs for setup, etc (very generous)

for pop_model in "rind" "uind" "eBN" "rBN" "eBNc" "rBNc"
do
  for classifier in "NN_V2H1" "NN_V2H2" "NN_V3H2"
  do
    timeout "$HARD_TIMEOUT" python -u "$HERE/verify.py" \
      --population-model "$pop_model" \
      --classifier "$classifier" \
      --no-qual \
      --timeout "$TIMEOUT" \
      "$@" \
      | tee "$OUT_DIR/${pop_model}_${classifier}_no_qual.log"
    timeout "$TIMEOUT" python -u "$HERE/verify.py" \
      --population-model "$pop_model" \
      --classifier "$classifier" \
      --qual \
      --timeout "$TIMEOUT" \
      "$@" \
      | tee "$OUT_DIR/${pop_model}_${classifier}_qual.log"
  done
done

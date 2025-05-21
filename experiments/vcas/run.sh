#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

TIMEOUT=600
OUT_DIR="$HERE/../output/$TIMESTAMP/vcas"
mkdir -p "$OUT_DIR"
python -u "$HERE/safety.py" \
  --timeout "$TIMEOUT" "$@" \
  | tee "$OUT_DIR/vcas_1.log"

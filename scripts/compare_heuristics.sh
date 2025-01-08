#!/bin/bash

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
export TIMESTAMP

./experiments/fairsquare/run.sh --probability-bounds-config "$1"
# ./experiments/acasxu/compare_heuristics.sh --probability-bounds-config "$1"
# ./experiments/mini_acs_income/run_base.sh --probability-bounds-config "$1"

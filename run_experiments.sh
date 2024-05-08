#!/bin/bash

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
export TIMESTAMP

./experiments/fairsquare/run.sh "$@"
./experiments/acasxu/run.sh "$@"
./experiments/mini_acs_income/run.sh "$@"

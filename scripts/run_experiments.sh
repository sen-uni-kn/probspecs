#!/bin/bash

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
export TIMESTAMP

./experiments/fairsquare/run.sh "$@"
./experiments/acasxu/run.sh "$@"
# while the other tools we compare to scarcely log, PreimageApproxForNN
# logs extensively, so enable logging in probspecs as well, for better comparability.
./experiments/vcas/run.sh "$@" --log
./experiments/mini_acs_income/run.sh "$@"

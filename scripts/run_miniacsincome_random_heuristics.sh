#!/bin/bash

for i in {1..10}
do
  ./experiments/mini_acs_income/run_base.sh \
      --probability-bounds-config experiments/config/random.yaml \
      --random-seed "$i" \
      "$@"
done

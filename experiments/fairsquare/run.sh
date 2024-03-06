#!/bin/bash

HERE="$(dirname "$0")"

for pop_model in "ind" "BN" "BNc"
do
  for classifier in "NN_V2H1" "NN_V2H2" "NN_V3H2"
  do
    timeout 60s python -u "$HERE/verify.py" \
      --population-model "$pop_model" \
      --classifier "$classifier" \
      --no-qual \
      | tee "$HERE/../output/fairsquare_${pop_model}_${classifier}_no_qual.log"
    timeout 60s python -u "$HERE/verify.py" \
      --population-model "$pop_model" \
      --classifier "$classifier" \
      --qual \
      | tee "$HERE/../output/fairsquare_${pop_model}_${classifier}_qual.log"
  done
done

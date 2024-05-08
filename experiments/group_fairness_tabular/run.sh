#!/bin/bash

HERE="$(dirname "$0")"

if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

OUT_DIR="$HERE/../output/$TIMESTAMP/adult"
mkdir -p "$OUT_DIR"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Female \
  --advantaged-group Male \
  | tee "$OUT_DIR/demographic_parity_female_male.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Black \
  --advantaged-group White \
  | tee "$OUT_DIR/demographic_parity_black_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Asian-Pac-Islander \
  --advantaged-group White \
  | tee "$OUT_DIR/demographic_parity_asian_pac_islander_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Amer-Indian-Eskimo \
  --advantaged-group White \
  | tee "$OUT_DIR/demographic_parity_amer_indian_eskimo_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Race-Other \
  --advantaged-group White \
  | tee "$OUT_DIR/demographic_parity_other_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Non-White \
  --advantaged-group White \
  | tee "$OUT_DIR/demographic_parity_non_white_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Own-child \
  --advantaged-group Married \
  | tee "$OUT_DIR/demographic_parity_own_child_married.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Unmarried \
  --advantaged-group Married \
  | tee "$OUT_DIR/demographic_parity_unmarried_married.log"

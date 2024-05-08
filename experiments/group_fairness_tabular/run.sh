#!/bin/bash

HERE="$(dirname "$0")"

echo "Adult"
=======
if [ -z ${TIMESTAMP+x} ];
then
  TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
fi

OUT_DIR="$HERE/../output/$TIMESTAMP/adult"
mkdir -p "$OUT_DIR"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Female \
  --advantaged-group Male \
  $@ \
  | tee "$OUT_DIR/demographic_parity_female_male.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Black \
  --advantaged-group White \
  $@ \
  | tee "$OUT_DIR/demographic_parity_black_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Asian-Pac-Islander \
  --advantaged-group White \
  $@ \
  | tee "$OUT_DIR/demographic_parity_asian_pac_islander_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Amer-Indian-Eskimo \
  --advantaged-group White \
  $@ \
  | tee "$OUT_DIR/demographic_parity_amer_indian_eskimo_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Race-Other \
  --advantaged-group White \
  $@ \
  | tee "$OUT_DIR/demographic_parity_other_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Non-White \
  --advantaged-group White \
  $@ \
  | tee "$OUT_DIR/demographic_parity_non_white_white.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Own-child \
  --advantaged-group Married \
  $@ \
  | tee "$OUT_DIR/demographic_parity_own_child_married.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset Adult \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Unmarried \
  --advantaged-group Married \
  $@ \
  | tee "$OUT_DIR/_demographic_parity_unmarried_married.log"

echo "SouthGerman"

OUT_DIR="$HERE/../output/$TIMESTAMP/south_german"
mkdir -p "$OUT_DIR"

timeout 12h python -u "$HERE/verify.py" \
  --dataset SouthGerman \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Female \
  --advantaged-group Male \
  $@ \
  | tee "$OUT_DIR/demographic_parity_female_male.log"

timeout 4h python -u "$HERE/verify.py" \
  --dataset SouthGerman \
  --population-model bayes-net \
  --fairness-criterion demographic-parity \
  --network tiny_network \
  --disadvantaged-group Foreign-Worker \
  --advantaged-group Non-Foreign-Worker \
  $@ \
  | tee "$OUT_DIR/demographic_parity_foreign_worker.log"


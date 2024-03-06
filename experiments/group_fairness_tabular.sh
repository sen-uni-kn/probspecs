#!/bin/bash

EPATH="experiments/group_fairness_tabular"

timeout 4h python -u "$EPATH/verify" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Female \
  --advantaged-group Male \
  | tee "experiments/adult_demographic_parity_female_male.log"

timeout 4h python -u "$EPATH/verify" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Black \
  --advantaged-group White \
  | tee "experiments/adult_demographic_parity_black_white.log"

timeout 4h python -u "$EPATH/verify" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Asian-Pac-Islander \
  --advantaged-group White \
  | tee "experiments/adult_demographic_parity_asian_pac_islander_white.log"

timeout 4h python -u "$EPATH/verify" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Amer-Indian-Eskimo \
  --advantaged-group White \
  | tee "experiments/adult_demographic_parity_amer_indian_eskimo_white.log"

timeout 4h python -u "$EPATH/verify" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Race-Other \
  --advantaged-group White \
  | tee "experiments/adult_demographic_parity_other_white.log"

timeout 4h python -u "$EPATH/verify" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Non-White \
  --advantaged-group White \
  | tee "experiments/adult_demographic_parity_non_white_white.log"

timeout 4h python -u "$EPATH/verify" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Own-child \
  --advantaged-group Married \
  | tee "experiments/adult_demographic_parity_own_child_married.log"

timeout 4h python -u "$EPATH/verify" \
  --dataset Adult \
  --population-model bayesian-network \
  --fairness-criterion demographic-parity \
  --network tiny_network.pyt \
  --disadvantaged-group Unmarried \
  --advantaged-group Married \
  | tee "experiments/adult_demographic_parity_unmarried_married.log"

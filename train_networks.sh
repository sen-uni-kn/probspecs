#!/bin/bash

echo "Training Adult networks"
# full hyperparameter search for two hours
python experiments/group_fairness_tabular/train_network.py \
  --dataset Adult --out resources/adult/network.pyt \
  --save-hyperparameters resources/adult/network_hyperparameters.yaml \
  --trials None --timeout 2700 --jobs 4
# train a smaller network
python experiments/group_fairness_tabular/train_network.py \
  --dataset Adult --out resources/adult/small_network.pyt \
  --save-hyperparameters resources/adult/small_network_hyperparameters.yaml \
  --trials None --timeout 600 --jobs 4 \
  --architecture 20 10
python experiments/group_fairness_tabular/train_network.py \
  --dataset Adult --out resources/adult/tiny_network.pyt \
  --save-hyperparameters resources/adult/tiny_network_hyperparameters.yaml \
  --trials None --timeout 600 --jobs 4 \
  --architecture 10

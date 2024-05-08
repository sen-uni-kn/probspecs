#!/bin/bash

echo "Training Adult networks"
# full hyperparameter search for two hours
python experiments/train_on_tabular_datasets.py \
  --dataset Adult --out resources/adult/network.pyt \
  --save-hyperparameters resources/adult/network_hyperparameters.yaml \
  --trials None --timeout 2700 --jobs 4
# train a smaller network
python experiments/train_on_tabular_datasets.py \
  --dataset Adult --out resources/adult/small_network.pyt \
  --save-hyperparameters resources/adult/small_network_hyperparameters.yaml \
  --trials None --timeout 600 --jobs 4 \
  --architecture 20 10
python experiments/train_on_tabular_datasets.py \
  --dataset Adult --out resources/adult/tiny_network.pyt \
  --save-hyperparameters resources/adult/tiny_network_hyperparameters.yaml \
  --trials None --timeout 600 --jobs 4 \
  --architecture 10

echo "Training SouthGerman networks"
python experiments/train_on_tabular_datasets.py \
  --dataset SouthGerman --out resources/south_german/network.pyt \
  --save-hyperparameters resources/south_german/network_hyperparameters.yaml \
  --trials None --timeout 300 --jobs 4
python experiments/train_on_tabular_datasets.py \
  --dataset SouthGerman --out resources/south_german/tiny_network.pyt \
  --save-hyperparameters resources/south_german/tiny_network_hyperparameters.yaml \
  --trials None --timeout 60 --jobs 4 \
  --architecture 10

echo "Training MiniACSIncome networks"
for i in {1..8}
do
  NET_NAME="MiniACSIncome-${i}_network"
  python experiments/train_on_tabular_datasets.py \
    --dataset "MiniACSIncome-$i" \
    --out "resources/MiniACSIncome/${NET_NAME}.pyt" \
    --use-hyperparameters "resources/MiniACSIncome/${NET_NAME}_hyperparameters.yaml"
done

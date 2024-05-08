#!/bin/bash

echo "Training MiniACSIncome networks"
for i in {1..8}
do
  NET_NAME="MiniACSIncome-${i}_network"
  python experiments/train_on_tabular_datasets.py \
    --dataset "MiniACSIncome-$i" \
    --out "resources/MiniACSIncome/${NET_NAME}.pyt" \
    --use-hyperparameters "resources/MiniACSIncome/${NET_NAME}_hyperparameters.yaml"
done

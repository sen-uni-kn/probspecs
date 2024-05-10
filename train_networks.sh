#!/bin/bash

# echo "Training MiniACSIncome networks"
# for i in {1..8}
# do
#   NET_NAME="MiniACSIncome-${i}_network"
#   python experiments/train_on_tabular_datasets.py \
#     --dataset "MiniACSIncome-$i" \
#     --out "resources/MiniACSIncome/${NET_NAME}.pyt" \
#     --use-hyperparameters "resources/MiniACSIncome/${NET_NAME}_hyperparameters.yaml"
# done

# vary network size
# for size in 20 30 40 50 60 70 80 90 100 200 300 400 500 600 700 800 900 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
# do
#   NET_NAME="MiniACSIncome-4_size_${size}"
#   python experiments/train_on_tabular_datasets.py \
#     --dataset "MiniACSIncome-4" \
#     --out "resources/MiniACSIncome/${NET_NAME}.pyt" \
#     --use-hyperparameters "resources/MiniACSIncome/MiniACSIncome-4_size_hyperparameters.yaml" \
#     --architecture "$size"
# done

for depth in {2..10}
do
  NET_NAME="MiniACSIncome-4_depth_${depth}"
  python experiments/train_on_tabular_datasets.py \
    --dataset "MiniACSIncome-4" \
    --out "resources/MiniACSIncome/${NET_NAME}.pyt" \
    --use-hyperparameters "resources/MiniACSIncome/MiniACSIncome-4_size_hyperparameters.yaml" \
    --architecture $(printf '10 %.0s' $(seq 1 "$depth"))
done

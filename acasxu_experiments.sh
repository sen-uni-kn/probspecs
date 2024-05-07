#!/bin/bash
TIMEOUT=1800
for net in "2_1" "2_2" "2_3" "2_4" "2_5" "2_6" "2_7" "2_8" "2_9"
do
  python experiments/acasxu/safety.py --network "$net" --property 2 --timeout "$TIMEOUT" \
  | tee "experiments/output/acasxu_${net}_property2.txt"
done
for net in "1_9" "net_1_9_property_7_partially_repaired_1" "net_1_9_property_7_partially_repaired_2" "net_1_9_property_7_partially_repaired_3"
do
  python experiments/acasxu/safety.py --network "$net" --property 7 --timeout "$TIMEOUT" \
  | tee "experiments/output/acasxu_${net}_property7.txt"
done
for net in "2_9" "net_2_9_property_8_unknown"
do
  python experiments/acasxu/safety.py --network "$net" --property 8 --timeout "$TIMEOUT" \
  | tee "experiments/output/acasxu_${net}_property8.txt"
done


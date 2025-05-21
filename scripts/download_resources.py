#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
from experiments.mini_acs_income import MiniACSIncome
from experiments.utils import get_acasxu_network, get_vcas_network

if __name__ == "__main__":
    MiniACSIncome(".datasets", num_variables=8, download=True)

    for i0 in range(2, 6):
        for i1 in range(1, 10):
            get_acasxu_network(i0, i1)

    get_vcas_network(1)

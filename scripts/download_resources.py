#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import miniacsincome
from experiments.utils import get_acasxu_network, get_vcas_network

if __name__ == "__main__":
    miniacsincome.download_all()

    for i0 in range(2, 6):
        for i1 in range(1, 10):
            get_acasxu_network(i0, i1)

    get_vcas_network(1)

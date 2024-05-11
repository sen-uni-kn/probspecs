#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
from experiments.mini_acs_income import MiniACSIncome

if __name__ == "__main__":
    MiniACSIncome("../.datasets", num_variables=8, download=True)

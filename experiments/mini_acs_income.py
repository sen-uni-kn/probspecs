# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from collections import defaultdict
from pathlib import Path
from typing import Union
import os

import numpy as np
import folktables
import torch
from torch.utils.data import Dataset

from probspecs import TabularInputSpace


class MiniACSIncome(Dataset):
    """
    A variant of the `ACSIncome` dataset from Ding et al. [1].
    This dataset uses fewer variables (the precise number can be configured) to facilitate scalability experiments
    on the size of the input space.
    Additionally, we use the binary sensitive attribute `SEX`, as recorded in the ACS data.

    Sizes:
     - 1 features: 2 inputs, 2 effective values
     - 2 features: 10 inputs, 10 total discrete values
     - 3 features: 34 inputs, 34 total discrete values
     - 4 features: 35 inputs, 134 total discrete values
     - 5 features: 40 inputs, 139 total discrete values
     - 6 features: 49 inputs, 148 total discrete values
     - 7 features: 67 inputs, 166 total discrete values
     - 8 features: 68 inputs, 266 total discrete values

    [1]: Frances Ding, Moritz Hardt, John Miller, Ludwig Schmidt:
    Retiring Adult: New Datasets for Fair Machine Learning. NeurIPS 2021: 6478-6490
    """

    survey_year = "2018"
    horizon = "1-Year"
    intended_size = 100000
    seed = 858460345379175
    variables_order = (
        "SEX",  # target sensitive attribute
        "COW",  # likely predictive and few values
        "SCHL",  # likely predictive, few values
        "WKHP",  # likely predictive, moderately many values
        "MAR",  # other sensitive attributes, ordered by number of values
        "RAC1P",
        "RELP",
        "AGEP",
        # The two below do not have category definitions
        # "POBL",  # attributes with many values (> 100)
        # "OCCP",
    )

    def __init__(
        self,
        root: Union[str, os.PathLike],
        num_variables: int = 3,
        download: bool = False,
        normalize: bool = False,
    ):
        """
        Creates a new :code:`MiniACSIncome` dataset.
        Uses the ACS person survey data from 2018 for the 1-Year time horizon.

        :param root: The root directory containing the data.
         If :code:`download=True` and the data isn't present in :code:`root`,
         it is downloaded to :code:`root`.
        :param num_variables: How many variables to include. From 1 to 10.
         The size of the input space grows non-linearly with the number of variables.
        :param download: Whether to download the dataset if it is not
          present in the :code:`root` directory.
        :param normalize: Whether to z-score normalize the numeric variables.
        """
        self.root = Path(root) / "ACS"
        self.root.mkdir(parents=True, exist_ok=True)

        data_source = folktables.ACSDataSource(
            survey_year=self.survey_year,
            horizon=self.horizon,
            survey="person",
            root_dir=self.root,
        )
        total_size = 3236200  # a little less
        # due to the adult filter applied later on, we need to massively oversample
        density = 2 * self.intended_size / total_size
        acs_data = data_source.get_data(
            density=density,
            random_seed=self.seed,
            download=download,
        )
        acs_definitions = data_source.get_definitions(download=True)

        self.variables = list(self.variables_order[:num_variables])
        problem = folktables.BasicProblem(
            features=self.variables,
            target="PINCP",
            target_transform=lambda x: x > 50000,  # as ACSIncome
            group="SEX",
            preprocess=folktables.adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )
        features_with_defs = [
            feat for feat in problem.features if feat not in ("POBL", "OCCP")
        ]
        categories = folktables.generate_categories(
            features_with_defs, definition_df=acs_definitions
        )
        data, targets, groups = problem.df_to_pandas(
            acs_data, categories=categories, dummies=True
        )

        column_order = []
        for var in self.variables:
            if var in categories:
                for _, value in categories[var].items():
                    col_name = f"{var}_{value}"
                    if col_name in data.columns:
                        column_order.append(col_name)
            else:
                column_order.append(var)
        data = data[column_order]

        self.columns = data.columns
        self.data = torch.as_tensor(
            data.astype(float).to_numpy(copy=True), dtype=torch.float
        )
        self.targets = torch.as_tensor(
            targets.astype(int).to_numpy(copy=True), dtype=torch.long
        ).squeeze()
        self.groups = torch.as_tensor(
            groups.astype(int).to_numpy(copy=True), dtype=torch.long
        ).squeeze()

        if normalize:
            means = self.data.mean(dim=0)
            stds = self.data.std(dim=0)
            for var in ("WKHP", "AGEP"):
                if var in self.variables:
                    col = [i for i, col in enumerate(self.columns) if col == var][0]
                    self.data[:, col] = (self.data[:, col] - means[col]) / stds[col]

    @property
    def input_space(self):
        integer_ranges = {}
        if "AGEP" in self.variables:
            integer_ranges["AGEP"] = (17, 95)
        if "WKHP" in self.variables:
            integer_ranges["WKHP"] = (1, 99)
        categorical_values = defaultdict(list)
        for column in self.columns:
            var, _, value = column.partition("_")
            categorical_values[var].append(value)
        categorical_values = {
            var: tuple(vals) for var, vals in categorical_values.items()
        }
        return TabularInputSpace(
            attributes=tuple(self.variables),
            data_types={
                var: TabularInputSpace.AttributeType.INTEGER
                if var in ("WKHP", "AGEP")
                else TabularInputSpace.AttributeType.CATEGORICAL
                for var in self.variables
            },
            continuous_ranges={},
            integer_ranges=integer_ranges,
            categorical_values=categorical_values,
        )

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    # def __getitems__(self, items):
    #     return self.data[items], self.targets[items]

    def __len__(self):
        return self.data.size(0)

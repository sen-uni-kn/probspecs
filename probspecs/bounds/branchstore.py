# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from collections import OrderedDict

import torch


class BranchStore:
    """
    A data structure for storing branches in branch and bound
    A branch consists of an input domain and output bounds.
    Both are represented by a pair of lower bounds (lb) and
    upper bounds (ub).
    All values are stored in tensors to facilitate fast batch
    processing.

    BranchStores support sorting their contents using the :code:`sort`
    method.
    Before sorting, branches are stored in order of addition.
    If further branches are added after sorting, they are appended at the
    end of the sorted branch data structure.
    """

    def __init__(
        self,
        in_shape: tuple | torch.Size,
        out_shape: tuple | torch.Size,
        **further_shapes: tuple | torch.Size,
    ):
        """
        Create an empty :class:`BranchStore`.

        :param in_shape: The shape of the input.
        :param out_shape: The shape of the output.
        :param further_shapes: Shapes of further values to store in
         this :class:`BranchStore`.
         The keywords you use for the shapes are the keys for which
         the corresponding values are stored.
        """
        self.__data = OrderedDict()
        self.__data["in_lbs"] = torch.empty((0,) + in_shape)
        self.__data["in_ubs"] = torch.empty((0,) + in_shape)
        self.__data["out_lbs"] = torch.empty((0,) + out_shape)
        self.__data["out_ubs"] = torch.empty((0,) + out_shape)
        for key, shape in further_shapes.items():
            self.__data[key] = torch.empty((0,) + shape)
        self.__permute = torch.empty((0,), dtype=torch.long)

    def append(
        self,
        in_lbs: torch.Tensor,
        in_ubs: torch.Tensor,
        out_lbs: torch.Tensor,
        out_ubs: torch.Tensor,
        **further_values: torch.Tensor,
    ):
        """
        Add a batch of branches to this data structure.

        :param in_lbs: The lower bounds of the input domains of the branches.
        :param in_ubs: The upper bounds of the input domains of the branches.
        :param out_lbs: The lower bounds of the output for the branches.
        :param out_ubs: The upper bounds of the output for the branches.
        :param further_values: Further values to store for the branches.
        """
        old_len = len(self)
        further_values.update(
            {
                "in_lbs": in_lbs,
                "in_ubs": in_ubs,
                "out_lbs": out_lbs,
                "out_ubs": out_ubs,
            }
        )
        for key, values in further_values.items():
            if values.ndim < self.__data[key].ndim:
                values = values.unsqueeze(0)  # add batch dimension
            self.__data[key] = torch.vstack([self.__data[key], values])
        new_idx = torch.arange(start=old_len, end=len(self), dtype=torch.long)
        self.__permute = torch.hstack([self.__permute, new_idx])

    def extend(self, other: "BranchStore"):
        """
        Augment this branch store with the branches from another branch store.
        """
        if set(self.__data.keys()) != set(other.__data.keys()):
            raise ValueError(
                "Other branch stores does not store the same "
                f"values as this branch store."
                f"Other: {set(other.__data.keys())}; self: {set(self.__data.keys())}"
            )
        self.append(**other.__data)

    def sort(self, scores: torch.Tensor, descending=True, stable=False):
        """
        Sorts the branches according to the :code:`scores`.

        :param scores: A score for each branch (a vector).
         This is used as the key for sorting.
        :param descending: Sort for descending scores (branch with highest score
         is first branch)
        :param stable: Sort stably (preserve order of equivalent values).
        """
        # argsort has no argument "stable"
        _, self.__permute = torch.sort(scores, descending=descending, stable=stable)

    def pop(self, n: int) -> "BranchStore":
        """
        Removes and returns the first :code:`n` values in this branch store.

        :param n: The number of branches to retrieve.
        :return: A new branch store containing the removed values.
        """
        select_idx = self.__permute[:n]
        retain_idx = self.__permute[n:]

        further_shapes = OrderedDict(
            [
                (key, values.shape[1:])
                for key, values in self.__data.items()
                if key not in ("in_lbs", "in_ubs", "out_lbs", "out_ubs")
            ]
        )
        selected_store = BranchStore(
            in_shape=self.input_shape, out_shape=self.output_shape, **further_shapes
        )
        for key, values in self.__data.items():
            selected_store.__data[key] = values.index_select(0, select_idx)
            self.__data[key] = values.index_select(0, retain_idx)
        # index_select with permute has sorted the values
        selected_store.__permute = torch.arange(len(select_idx))
        self.__permute = torch.arange(len(retain_idx))
        return selected_store

    def __getitem__(
        self, item
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ...]:
        """
        Returns input lower bounds, input upper bounds, output lower bounds,
        output upper bounds and further stored values for the requested index
        (or slice).
        """
        indices = self.__permute[item]
        return tuple(values.index_select(0, indices) for values in self.__data.values())

    def __getattr__(self, item):
        try:
            return self.__data[item].index_select(0, self.__permute)
        except KeyError:
            raise AttributeError()

    @property
    def input_shape(self) -> torch.Size:
        return self.shape("in_lbs")

    @property
    def output_shape(self) -> torch.Size:
        return self.shape("out_lbs")

    def shape(self, key) -> torch.Size:
        """
        The shape of the values sorted for :code:`key`.
        This key may be any of the keywords of the further shapes you
        supply at initialisation.
        """
        return self.__data[key].shape[1:]

    def __len__(self):
        return self.__data["in_lbs"].size(0)

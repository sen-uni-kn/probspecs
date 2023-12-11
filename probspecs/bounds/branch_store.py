# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from collections import OrderedDict
from math import prod

import torch


class BranchStore:
    """
    A data structure for storing branches in branch and bound
    A branch (normally) consists of an input domain and output bounds.
    Both are represented by a pair of lower bounds (lb) and
    upper bounds (ub).
    Branches can also store further arbitrary values.
    All values are stored in tensors to facilitate fast batch
    processing.

    All values stored in a branch store are available as attributes.
    For example, the input lower bounds of a :class:`BranchStore` :code:`store`
    can be accessed as :code:`store.in_lbs`.
    Similarly, :code:`store.in_ubs`, :code:`store.out_lbs` and :code:`store.out_ubs`
    are available.
    Further values stored in a :class:`BranchStore` are available similarly.

    BranchStores support sorting their contents using the :code:`sort`
    method.
    Before sorting, branches are stored in order of addition.
    If further branches are added after sorting, they are appended at the
    end of the sorted branch data structure.
    """

    def __init__(
        self,
        in_shape: tuple | torch.Size = None,
        out_shape: tuple | torch.Size = None,
        device: str | torch.device | None = None,
        **further_shapes: tuple | torch.Size,
    ):
        """
        Create an empty :class:`BranchStore`.

        :param in_shape: The shape of the input.
         If omitted, the attributes :code:`in_lbs` and :code:`in_ubs`
         are unavailable.
        :param out_shape: The shape of the output.
         If omitted, the attributes :code:`out_lbs` and :code:`out_ubs`
         are unavailable.
        :param device: Where to store tensors.
        :param further_shapes: Shapes of further values to store in
         this :class:`BranchStore`.
         The keywords you use for the shapes are the keys for which
         the corresponding values are stored.
        """
        self.__data = OrderedDict()
        if in_shape is not None:
            self.__data["in_lbs"] = torch.empty((0,) + in_shape, device=device)
            self.__data["in_ubs"] = torch.empty((0,) + in_shape, device=device)
        if out_shape is not None:
            self.__data["out_lbs"] = torch.empty((0,) + out_shape, device=device)
            self.__data["out_ubs"] = torch.empty((0,) + out_shape, device=device)
        for key, shape in further_shapes.items():
            self.__data[key] = torch.empty((0,) + shape, device=device)

    def append(
        self,
        **values: torch.Tensor,
    ):
        """
        Add a batch of branches to this data structure.

        :param values: The values to store for the branches.
         Examples include,
         - in_lbs: The lower bounds of the input domains of the branches.
         - in_ubs: The upper bounds of the input domains of the branches.
         - out_lbs: The lower bounds of the output for the branches.
         - out_ubs: The upper bounds of the output for the branches.
         - values for any further shapes you specified when initialising
           this branch store.
        """
        for key, values in values.items():
            if values.ndim < self.__data[key].ndim:
                values = values.unsqueeze(0)  # add batch dimension
            self.__data[key] = torch.vstack([self.__data[key], values])

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

    def drop(self, mask: torch.Tensor):
        """
        Drops branches where :code:`mask` is :code:`True`.

        :param mask: A boolean vector indicating which branches to drop.
        """
        mask = mask.squeeze()
        for key, values in self.__data.items():
            # values may also be just a vector...
            flat_values = values.reshape(-1, prod(values.shape[1:]))
            keep = flat_values[~mask, :]
            self.__data[key] = keep.reshape(-1, *values.shape[1:])

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
        _, permute = torch.sort(scores, descending=descending, stable=stable)
        for key, values in self.__data.items():
            self.__data[key] = values.index_select(0, permute)

    def pop(self, n: int) -> "BranchStore":
        """
        Removes and returns the first :code:`n` values in this branch store.

        :param n: The number of branches to retrieve.
        :return: A new branch store containing the removed values.
        """
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
            selected_store.__data[key] = values[:n]
            self.__data[key] = values[n:]
        return selected_store

    def __getitem__(self, item) -> tuple[torch.Tensor, ...]:
        """
        Returns the values of the item-th branch (item may also be a slice).
        When input and output shapes are supplied at initialisation,
        these values are the input lower bounds, input upper bounds, output lower bounds,
        output upper bounds and further values for which shapes were supplied on
        initialisation.
        """
        return tuple(values[item] for values in self.__data.values())

    def __getattr__(self, item):
        try:
            return self.__data[item]
        except KeyError:
            raise AttributeError()

    @property
    def input_shape(self) -> torch.Size:
        return self.shape("in_lbs") if "in_lbs" in self.__data else None

    @property
    def output_shape(self) -> torch.Size:
        return self.shape("out_lbs") if "out_lbs" in self.__data else None

    def shape(self, key) -> torch.Size:
        """
        The shape of the values sorted for :code:`key`.
        This key may be any of the keywords of the further shapes you
        supply at initialisation.
        """
        return self.__data[key].shape[1:]

    def __len__(self):
        return self.__data["in_lbs"].size(0)

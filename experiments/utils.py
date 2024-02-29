#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import os
import argparse

import numpy as np
import torch
from torch import nn

from probspecs.population_models import Normalize, Denormalize, Identity


@torch.no_grad()
def load_nnet(
    path: os.PathLike,
) -> tuple[nn.Sequential, tuple[torch.Tensor, torch.Tensor]]:
    """
    Loads a ReLU activated neural network model from a `.nnet` file.

    :param path: The path to the `.nnet` file
    :return: The neural network and the input space bounds from the `.nnet` file
    """
    with open(path) as f:
        line = f.readline()
        count = 1
        while line[0:2] == "//":
            line = f.readline()
            count += 1
        # num_layers doesn't include the inputs module!
        num_layers, input_size, output_size, _ = [
            int(x) for x in line.strip().split(",")[:-1]
        ]
        line = f.readline()

        # inputs module size, layer1size, layer2size...
        layer_sizes = [int(x) for x in line.strip().split(",")[:-1]]

        # the next line contains a flag that is not use; ignore
        f.readline()
        # symmetric = int(line.strip().split(",")[0])

        line = f.readline()
        input_minimums = [float(x) for x in line.strip().split(",") if x != ""]
        while len(input_minimums) < input_size:
            input_minimums.append(min(input_minimums))
        input_minimums = torch.as_tensor(input_minimums)

        line = f.readline()
        input_maximums = [float(x) for x in line.strip().split(",") if x != ""]
        while len(input_maximums) < input_size:
            input_maximums.append(max(input_maximums))
        input_maximums = torch.as_tensor(input_maximums)

        line = f.readline()
        means = [float(x) for x in line.strip().split(",")[:-1]]
        # if there are too little means given (we also need one for the output)
        # fill up with 0, which will cause no modifications in the data
        if len(means) < input_size + 1:
            means.append(0.0)
        means = torch.tensor(means)

        line = f.readline()
        ranges = [float(x) for x in line.strip().split(",")[:-1]]
        # same as with means
        if len(ranges) < input_size + 1:
            ranges.append(1.0)
        ranges = torch.tensor(ranges)

        weights = []
        biases = []

        # Matrix of Neural Network
        #
        # The first dimension will be the module number
        # The second dimension will be 0 for weights, 1 for biases
        # The third dimension will be the number of neurons in that module
        # The fourth dimension will be the number of inputs to that module
        #
        # Note that the bias array will have only number per neuron, so
        # its fourth dimension will always be one
        for layer_idx in range(num_layers):
            previous_layer_size = layer_sizes[layer_idx]
            current_layer_size = layer_sizes[layer_idx + 1]

            weights.append([])
            biases.append([])

            weights[layer_idx] = np.zeros((current_layer_size, previous_layer_size))

            for i in range(current_layer_size):
                line = f.readline()
                aux = [float(x) for x in line.strip().split(",")[:-1]]

                for j in range(previous_layer_size):
                    weights[layer_idx][i, j] = aux[j]

            # biases
            biases[layer_idx] = np.zeros(current_layer_size)
            for i in range(current_layer_size):
                line = f.readline()
                x = float(line.strip().split(",")[0])
                biases[layer_idx][i] = x

    modules = []
    for i in range(num_layers - 1):  # hidden layers
        linear_layer = nn.Linear(
            in_features=layer_sizes[i],  # layer_sizes[0] contains the number of inputs
            out_features=layer_sizes[i + 1],
        )
        # torch wants shape units x inputs, which is what we already have
        linear_layer.weight.data = torch.as_tensor(weights[i]).to(linear_layer.weight)
        linear_layer.bias.data = torch.as_tensor(biases[i]).to(linear_layer.bias)
        modules.append(linear_layer)
        modules.append(nn.ReLU())
    # add the output layer
    output_layer = nn.Linear(in_features=layer_sizes[-2], out_features=layer_sizes[-1])
    output_layer.weight.data = torch.as_tensor(weights[-1]).to(output_layer.weight)
    output_layer.bias.data = torch.as_tensor(biases[-1]).to(output_layer.bias.data)
    modules.append(output_layer)

    normalize_input = Normalize(means[:-1].unsqueeze(0), ranges[:-1].unsqueeze(0))
    denormalize_output = Denormalize(means[-1].reshape(1, 1), ranges[-1].reshape(1, 1))
    return nn.Sequential(normalize_input, *modules, denormalize_output), (
        input_minimums,
        input_maximums,
    )
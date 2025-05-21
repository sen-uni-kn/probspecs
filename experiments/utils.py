#  Copyright (c) 2024. David Boetius
#  Licensed under the MIT License
import os
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np
import torch
import psutil
import cpuinfo
from GPUtil import GPUtil
from torch import nn
import requests
import onnx
from onnx2pytorch import ConvertModel

from probspecs.population_models import Normalize, Denormalize


def get_acasxu_network(
    i1: int,
    i2: int,
    root="resources/acasxu",
    base_url="https://raw.githubusercontent.com/guykatzz/ReluplexCav2017/60b482eec832c891cb59c0966c9821e40051c082/nnet/",
) -> tuple[nn.Sequential, tuple[torch.Tensor, torch.Tensor]]:
    """
    Load an ACAS Xu network from the `root` directory.
    Download the network from the Reluplex GitHub repository if necessary.
    """
    i1, i2 = int(i1), int(i2)
    if i1 < 1 or i1 > 5:
        raise ValueError("The first network index must be between 1 and 5")
    if i2 < 1 or i2 > 9:
        raise ValueError("The second network index must be between 1 and 9")

    net_file = f"ACASXU_run2a_{i1}_{i2}_batch_2000.nnet"
    root_dir = Path(root)
    target_file = root_dir / net_file
    if not target_file.exists():
        url = base_url + net_file
        print(f"Downloading ACAS Xu network {i1}, {i2} from {url}.")
        result = requests.get(url)
        if not result.ok:
            raise ValueError(
                f"Failed to download ACAS Xu network {i1}, {i2} from {url}."
            )

        root_dir.mkdir(exist_ok=True)
        target_file.touch()
        with open(target_file, "wb") as file:
            file.write(result.content)

    return load_nnet(target_file)


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


def get_vcas_network(
    i: int,
    root="resources/vcas",
    base_url=" https://raw.githubusercontent.com/Zhang-Xiyue/PreimageApproxForNNs/2cc0dc47e447b83f1e18626272a75d5e16059f12/model_dir/",
) -> nn.Module:
    """
    Load a VCAS network from the `root` directory.
    Download the network from the PreimageApproxForNNs GitHub repository if necessary.
    """
    i = int(i)
    if i < 1 or i > 9:
        raise ValueError("The network index must be between 1 and 9")

    net_file = f"VertCAS_{1}.onnx"
    root_dir = Path(root)
    target_file = root_dir / net_file
    if not target_file.exists():
        url = base_url + net_file
        print(f"Downloading VCAS network {i} from {url}.")
        result = requests.get(url)
        if not result.ok:
            raise ValueError(f"Failed to download VCAS network {i} from {url}.")

        root_dir.mkdir(exist_ok=True)
        target_file.touch()
        with open(target_file, "wb") as file:
            file.write(result.content)

    onnx_network = onnx.load(target_file)
    return ConvertModel(onnx_network)


def log_machine_and_code_details():
    """
    Collects and prints relevant experiment statistics:
     * time
     * git commit
     * platform details (includes operation system name)
     * processor name
     * installed memory
     * current memory usage
     * installed swap
     * current swap usage
     * number of GPus
     * for each GPU: name, memory, memory usage.
    """
    # try to get the current git commit
    # solution form https://stackoverflow.com/a/68215738/10550998
    # by Naelson Douglas
    git_folder = Path(".git")
    if git_folder.exists():
        head_name = (git_folder / "HEAD").read_text().split("\n")[0].split(" ")[-1]
        commit = (git_folder / head_name).read_text().replace("\n", "")
    else:
        commit = "unknown"
    stats = (
        f"Setup Details\n"
        f"----------------------------------------------------------------------\n"
        f"time: {datetime.now(timezone.utc)}\n"
        f"code version: {commit}\n"
        f"platform: {platform.platform(aliased=True)}\n"
        f"CPU:\n"
        f"  name: {cpuinfo.get_cpu_info()['brand_raw']}\n"
        f"  physical cores: {psutil.cpu_count(logical=False)}\n"
        f"  logical cores: {psutil.cpu_count(logical=True)}\n"
    )
    memory_stats = psutil.virtual_memory()
    stats += (
        f"memory:\n"
        f"  total: {memory_stats.total / (1024 ** 3):.2f} GB\n"
        f"  used: {memory_stats.percent}%\n"
    )
    swap_stats = psutil.swap_memory()
    stats += (
        f"swap:\n"
        f"    total: {swap_stats.total / (1024 ** 3):.3f} GB\n"
        f"    used: {swap_stats.percent}%\n"
    )
    gpus = GPUtil.getGPUs()
    if len(gpus) == 0:
        stats += "GPUs: 0\n"
    else:
        stats += f"GPUs: {len(gpus)}\n"
        for gpu in gpus:
            stats += (
                f"GPU {gpu.id}:\n"
                f"    name: {gpu.name}\n"
                f"    total GPU memory: {gpu.memoryTotal} MB\n"
                f"    used GPU memory: {100*gpu.memoryUsed/gpu.memoryTotal:.1f}%\n"
            )
    print(stats)

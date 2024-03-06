# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


if __name__ == "__main__":
    # Small net for tests
    torch.manual_seed(177878981064172)
    net = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, stride=3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(9 * 9 * 2, 10),
    )
    optim = torch.optim.Adam(net.parameters())
    epochs = 10
    batch_size = 64
    large_batch_size = 1024

    train_set = MNIST("../.datasets", transform=ToTensor(), download=True)
    test_set = MNIST("../.datasets", train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(train_set, large_batch_size)

    loss_func = nn.CrossEntropyLoss()

    def batch_accuracy(inputs, targets):
        predictions = net(inputs)
        return (torch.argmax(predictions, dim=1) == targets).float().mean()

    def accuracy(loader):
        result = 0.0
        for inputs, targets in loader:
            result += batch_accuracy(inputs, targets)
        return result / len(loader)

    log_frequency = len(train_loader) // 10
    for epoch in range(epochs):
        for i in range(len(train_loader)):
            optim.zero_grad()

            inputs, targets = next(iter(train_loader))
            predictions = net(inputs)
            loss = loss_func(predictions, targets)
            loss.backward()
            optim.step()

            if i % log_frequency == 0 or i == len(train_loader) - 1:
                batch_acc = batch_accuracy(inputs, targets)
                train_acc = accuracy(train_loader)
                test_acc = accuracy(test_loader)
                print(
                    f"Epoch {epoch:3.0f}/{epochs} ({i/(len(train_loader)-1)*100:3.0f}%), "
                    f"Batch Loss: {loss:6.4f}, Batch Accuracy: {batch_acc*100:4.1f}%, "
                    f"Train Accuracy: {train_acc*100:4.1f}%, "
                    f"Test Accuracy: {test_acc*100:4.1f}%"
                )
    torch.save(net, "mnist_network.pyt")

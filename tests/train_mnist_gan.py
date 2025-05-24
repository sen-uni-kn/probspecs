# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torchvision.utils as vutils


if __name__ == "__main__":
    # Train a MNIST WassersteinGAN.
    # Based on:
    #  - https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
    #  - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    # Install tensorboard to view the training progress
    device = "cuda:0"

    # # Small conv transpose generator for tests
    # torch.manual_seed(659917698158452)
    # generator = nn.Sequential(
    #     nn.ConvTranspose2d(4, 49, kernel_size=4, stride=1, bias=False),  # 49 x 4 x 4
    #     nn.BatchNorm2d(49, affine=True),
    #     nn.LeakyReLU(negative_slope=0.2),
    #     nn.ConvTranspose2d(49, 12, kernel_size=4, stride=4, bias=False),  # 12 x 16 x 16
    #     nn.BatchNorm2d(12, affine=True),
    #     nn.LeakyReLU(negative_slope=0.2),
    #     nn.ConvTranspose2d(12, 1, kernel_size=13, stride=1, bias=False),  # 1 x 28 x 28
    #     nn.Sigmoid(),
    # )
    # discriminator = nn.Sequential(  # in: 1 x 28 x 28
    #     nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=3, bias=False),  # 8 x 16 x 16
    #     nn.BatchNorm2d(8),
    #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #     nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16 x 8 x 8
    #     nn.BatchNorm2d(16),
    #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #     nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 16 x 4 x 4
    #     nn.BatchNorm2d(32),
    #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64 x 2 x 2
    #     nn.BatchNorm2d(64),
    #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #     nn.Flatten(),  # 256
    #     nn.Linear(256, 128),
    #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #     nn.Linear(128, 2),
    # )
    # generator = generator.to(device)
    # discriminator = discriminator.to(device)
    # gen_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
    # gen_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     gen_optim, [10, 40]  # epochs
    # )
    # disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    # disc_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(disc_optim, [5, 25])
    # epochs = 60
    # discriminator_updates = 3
    # batch_size = 512
    # disc_label_noise = 0.1

    # Small fully connected generator for tests
    torch.manual_seed(659917698158452)

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(4, 196)
            self.bn1 = nn.BatchNorm1d(196)
            self.lin2 = nn.Linear(196, 392)
            self.bn2 = nn.BatchNorm1d(392)
            self.lin3 = nn.Linear(392, 784)
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        def __call__(self, x):
            x = x.flatten(1)
            for lin, bn in zip([self.lin1, self.lin2], [self.bn1, self.bn2]):
                x = lin(x)
                x = bn(x.unsqueeze(-1)).squeeze(-1)
                x = self.leaky_relu(x)
            x = self.lin3(x)
            x = torch.sigmoid(x)
            return torch.reshape(x, (-1, 1, 28, 28))

    generator = Generator()
    discriminator = nn.Sequential(  # in: 1 x 28 x 28
        nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=3, bias=False),  # 8 x 16 x 16
        nn.BatchNorm2d(8),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),  # 16 x 8 x 8
        nn.BatchNorm2d(16),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 16 x 4 x 4
        nn.BatchNorm2d(32),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 64 x 2 x 2
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Flatten(),  # 256
        nn.Linear(256, 128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Linear(128, 2),
    )
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    gen_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
    gen_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(gen_optim, [25])  # epochs
    disc_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    disc_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(disc_optim, [5, 25])
    epochs = 50
    discriminator_updates = 2
    batch_size = 512
    disc_label_noise = 0.1

    real_data = MNIST("../.datasets", transform=ToTensor(), download=True)
    real_loader = DataLoader(real_data, batch_size, shuffle=True)

    ce_loss = nn.CrossEntropyLoss()

    # for evaluating training progress
    fixed_noise = torch.randn((32, 4, 1, 1), device=device)
    tensorboard = SummaryWriter("../.tensorboard")
    fake_data = generator(fixed_noise).detach().cpu()
    image_grid = vutils.make_grid(fake_data, nrow=8)
    tensorboard.add_image("generated images", image_grid, global_step=0)

    num_iters = len(real_loader) // discriminator_updates
    log_frequency = num_iters // 10
    label = torch.empty((batch_size,), dtype=torch.long, device=device)
    label_flip_prob = torch.full((batch_size,), disc_label_noise, device=device)
    for epoch in range(epochs):
        for i in range(num_iters):
            disc_error_real = disc_error_fake = 1.0
            for i_disc in range(discriminator_updates):
                disc_optim.zero_grad()
                real_data, _ = next(iter(real_loader))
                real_data = real_data.to(device)
                label = 1 - torch.bernoulli(label_flip_prob).long()

                output = discriminator(real_data)
                disc_error_real = ce_loss(output, label)
                disc_error_real.backward()

                with torch.no_grad():
                    noise = torch.randn((batch_size, 4, 1, 1), device=device)
                    fake_data = generator(noise)
                    label = torch.bernoulli(label_flip_prob).long()

                output = discriminator(fake_data)
                disc_error_fake = ce_loss(output, label)
                disc_error_fake.backward()  # accumulates gradients

                disc_optim.step()
                # clip discriminator weights
                with torch.no_grad():
                    for layer in discriminator:
                        if isinstance(layer, (nn.Conv2d, nn.Linear)):
                            layer.weight.clip(-0.01, 0.01)

            gen_optim.zero_grad()
            noise = torch.randn((batch_size, 4, 1, 1), device=device)
            fake_data = generator(noise)
            label.fill_(1)  # fake data is real for generator loss
            score = discriminator(fake_data)
            gen_error = ce_loss(score, label)
            gen_error.backward()
            gen_optim.step()

            if i % log_frequency == 0 or i == num_iters - 1:
                disc_error = (disc_error_real + disc_error_fake) / 2
                print(
                    f"Epoch {epoch + 1:3.0f}/{epochs} ({i / (num_iters - 1) * 100:3.0f}%), "
                    f"Generator Loss: {gen_error:6.4f}, "
                    f"Discriminator Loss: {disc_error:6.4f} "
                    f"({disc_error_real:6.4f}/{disc_error_fake:6.4f}) (real/fake)"
                )
                step = epoch * num_iters + i + 1
                tensorboard.add_scalar("generator loss", gen_error, global_step=step)
                tensorboard.add_scalar(
                    "discriminator loss", disc_error, global_step=step
                )
                with torch.no_grad():
                    fake_data = generator(fixed_noise).detach().cpu()
                    image_grid = vutils.make_grid(fake_data, nrow=8)
                tensorboard.add_image("generated images", image_grid, global_step=step)
                tensorboard.flush()
        gen_lr_scheduler.step()
        disc_lr_scheduler.step()

    # torch.save(generator.to("cpu"), "mnist_generator.pyt")
    torch.save(generator.to("cpu").state_dict(), "mnist_generator_params.pyt")
    torch.save(discriminator.to("cpu"), "mnist_discriminator.pyt")

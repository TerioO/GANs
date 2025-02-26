import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import VisionDataset
from typing import Literal
import matplotlib.pyplot as plt
import numpy as np
from env import env
from tqdm import tqdm
import os
import shutil

def get_dataset(batch_size: int, transform: transforms):
    """
    Example:
        - datasets, dataloaders = prepare_dataset(32)
        - train_data, test_data = datasets
        - train_dataloader, test_dataloader = dataloaders
    """
    train_data = torchvision.datasets.MNIST(root=env["DATASET_DIR"],
                                            train=True,
                                            transform=transform,
                                            target_transform=None,
                                            download=True)
    test_data = torchvision.datasets.MNIST(root=env["DATASET_DIR"],
                                           train=False,
                                           transform=transform,
                                           target_transform=None,
                                           download=True)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=True)

    return [(train_data, test_data), (train_dataloader, test_dataloader)]

def print_about_dataset(dataset: torchvision.datasets.MNIST):
    print(f"[MNIST DATASET] {'-' * 200}")

    print(dataset)
    img, label = dataset[0]
    print(f"Nr. of samples: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    print(f"Classes to indexes: {dataset.class_to_idx}")
    print(f"Image[0] shape = {img.shape}")
    print(f"Image[0] label = {label}")

def print_visualize_random_images(dataset: torchvision.datasets.MNIST):
    fig = plt.figure(figsize=(12,9))
    for i in range(1,17):
        random_idx = torch.randint(0, len(dataset), size=[1]).item()
        img, label = dataset[random_idx]
        fig.add_subplot(4,4,i)
        plt.imshow(img.permute(1,2,0), cmap="gray")
        plt.title(dataset.classes[label])
        plt.axis(False)
    plt.show()

class Discrimintaor(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units,
        self.output_shape = output_shape
        self.disc = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units,
        self.output_shape = output_shape
        self.gen = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.Tanh()
        )

    def forward(self,x):
        return self.gen(x)

def train_loop(data_loader: DataLoader,
               epochs: int,
               generator: nn.Module,
               gen_optim: torch.optim.Optimizer,
               discriminator: nn.Module,
               disc_optim: torch.optim.Optimizer,
               criterion: nn.Module,
               device: str,
               writer_real: SummaryWriter,
               writer_fake: SummaryWriter):
    generator.to(device)
    discriminator.to(device)

    step = 0

    for epoch in tqdm(range(epochs)):
        for batch_idx, (img_real, img_real_label) in enumerate(data_loader):
            img_real = torch.as_tensor(img_real, device=device).view(-1, 28*28)
            img_real_label = torch.as_tensor(img_real_label, device=device)
            batch_size = img_real.shape[0]
            # print(img_real.shape)                  # [32,1,28,28]
            # print(img_real.view(-1, 28*28).shape)  # [32,784]
            # print(batch_size)                      # 32

            # Train discriminator
            noise = torch.randn(batch_size, generator.input_shape).to(device)
            fake = generator(noise)
            disc_real = discriminator(img_real).view(-1)
            disc_loss_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake).view(-1)
            disc_loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = (disc_loss_real + disc_loss_fake) / 2
            disc_optim.zero_grad()
            disc_loss.backward(retain_graph=True)
            disc_optim.step()

            # Train generator
            output = discriminator(fake).view(-1)
            gen_loss = criterion(output, torch.ones_like(output))
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            if batch_idx == 0:
                print(f"\nEpoch [{epoch}/{epochs}] | Loss D: {disc_loss:.4f} Loss G: {gen_loss:.4f}")

                with torch.inference_mode():
                    fake = generator(torch.randn((batch_size, generator.input_shape)).to(device)).reshape(-1, 1, 28, 28)
                    data = img_real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                    img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                    writer_fake.add_image("MNIST Fake Images", img_grid_fake, global_step=step)
                    writer_real.add_image("MNIST Real Images", img_grid_real, global_step=step)
                    step += 1

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    lr = 3e-4
    epochs = 50

    if os.path.exists("./logs"):
        shutil.rmtree("./logs")
    writer_fake = SummaryWriter(f"./logs/GAN_simple/fake")
    writer_real = SummaryWriter(f"./logs/GAN_simple/real")

    dataset, dataloader = get_dataset(batch_size, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]))
    train_data, test_data = dataset
    train_dataloader, test_dataloader = dataloader

    # print_about_dataset(train_data)
    # print_visualize_random_images(test_data)

    gen_0 = Generator(input_shape=64, hidden_units=256, output_shape=28*28*1)
    disc_0 = Discrimintaor(input_shape=28*28*1, hidden_units=256, output_shape=1)
    loss_fn = nn.BCELoss()
    gen_optim = torch.optim.Adam(gen_0.parameters(), lr=lr)
    disc_optim = torch.optim.Adam(disc_0.parameters(), lr=lr)

    train_loop(data_loader=train_dataloader,
               epochs=epochs,
               generator=gen_0,
               gen_optim=gen_optim,
               discriminator=disc_0,
               disc_optim=disc_optim,
               criterion=loss_fn,
               device=device,
               writer_real=writer_real,
               writer_fake=writer_fake)
main()
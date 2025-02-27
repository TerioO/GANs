import torch
import torchvision
import torchmetrics
from torch import nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from env import env
from tqdm import tqdm
import os
import shutil
import helpers

def train_CNN(model: nn.Module,
              epochs: int,
              dataloader: DataLoader,
              num_classes: int,
              loss_fn: nn.Module,
              optim: torch.optim,
              device: str,
              skip: bool = False):
    if skip:
        return

    json_data = helpers.read_json_log("cnn.json")
    results = []
    accuracy_fn = torchmetrics.classification.MulticlassAccuracy(
        num_classes=num_classes).to(device)
    model.train()

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = 0, 0
        for batch_idx, (img, label) in enumerate(dataloader):
            img = torch.as_tensor(img, device=device)
            label = torch.as_tensor(label, device=device)

            label_pred = model(img)
            loss = loss_fn(label_pred, label)
            train_loss += loss
            train_acc += accuracy_fn(label_pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        train_loss = train_loss/len(dataloader)
        train_acc = train_acc/len(dataloader)
        res = f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}"
        print(f"\nEpoch [{epoch+1}/{epochs}]")
        print(res)
        results.append(res)

    json_data["epochs"] += epochs
    json_data["results"] += results
    helpers.write_json_log("cnn.json", json_data)


class Discriminator(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gen(x)


def main():
    os.system("cls")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 2e-4
    epochs = 2
    batch_size = 32

    train, test, train_dataloader, test_dataloader = helpers.load_MNIST(
        transforms.ToTensor(), batch_size)

    helpers.write_json_log(filename="cnn.json",
                           json_obj={"epochs": 0, "results": []},
                           skip_if_exists=True)

    cnn = Discriminator(input_shape=28*28,
                        hidden_units=256,
                        output_shape=10).to(device)

    helpers.save_or_load_model(cnn, "cnn", "load")

    train_CNN(model=cnn,
              epochs=2,
              dataloader=train_dataloader,
              num_classes=len(train.classes),
              loss_fn=nn.CrossEntropyLoss(),
              optim=torch.optim.Adam(params=cnn.parameters(), lr=2e-4),
              device=device,
              skip=False)

    helpers.save_or_load_model(cnn, "cnn", "save")

    def make_predictions():
        print()

    def playground():
        img, label = next(iter(train_dataloader))
        print(img.shape)
        linear = nn.Linear(32, 256)
        flatten = nn.Flatten()
        output = flatten(img)
        print(output.shape)
        output = linear(img)
        print(output.shape)


    # playground()
main()

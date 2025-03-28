import math
from pathlib import Path
import random
import time
import typing
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
from torchsummary import summary
from typing import TypedDict, Dict
import torch.onnx

class IFilenames(TypedDict):
    dir: str
    generator: str
    discriminator: str
    gan: str
    tensorboard: str

class Discriminator(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.BatchNorm1d(num_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


def train_GAN(filenames: IFilenames,
              epochs: int,
              device: str,
              dataloader,
              gen: nn.Module,
              gen_optim: torch.optim.Optimizer,
              disc: nn.Module,
              disc_optim: torch.optim.Optimizer,
              criterion: nn.Module,
              skip: bool = False):
    """
    Paper: https://arxiv.org/pdf/1406.2661
    """
    try:
        if skip:
            return

        # Tensorboard init:
        writer = SummaryWriter(log_dir=filenames["tensorboard"],
                               filename_suffix=Path(filenames["tensorboard"]).name)

        # JSON init:
        json_log = helpers.read_json_log(filenames["dir"], filenames["gan"])
        results = []

        # Train time start:
        start_time = time.time()

        # Models init:
        gen.to(device)
        disc.to(device)
        gen.train()
        disc.train()
        for epoch in tqdm(range(epochs)):
            disc_epoch_loss, gen_epoch_loss = 0, 0
            disc_epoch_acc_real, disc_epoch_acc_fake = 0, 0
            for _batch_idx, (img, _) in enumerate(dataloader):
                img = torch.as_tensor(img, device=device)

                # Generate a fake image from random noise:
                input_noise = torch.randn(img.shape).to(device)
                img_fake = gen(input_noise)

                # Update discriminator weights:
                # y_pred_fake.shape = [batch_size, 1]
                y_pred_fake = disc(img_fake)
                # y_pred_real.shape = [batch_Size, 1]
                y_pred_real = disc(img)
                disc_epoch_acc_fake += y_pred_fake.mean().item()
                disc_epoch_acc_real += y_pred_real.mean().item()
                disc_loss_fake = criterion(y_pred_fake, torch.zeros(y_pred_fake.shape).to(device) * 0.1)
                disc_loss_real = criterion(y_pred_real, torch.ones(y_pred_real.shape).to(device) * 0.9)
                disc_loss = (disc_loss_fake + disc_loss_real) / 2
                disc_optim.zero_grad()
                disc_loss.backward(retain_graph=True)
                disc_optim.step()

                # Update generator weights:
                y_pred_fake = disc(img_fake)
                gen_loss = criterion(y_pred_fake, torch.ones(y_pred_fake.shape).to(device))
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()

                # Discriminator, Generator total loss:
                disc_epoch_loss += disc_loss
                gen_epoch_loss += gen_loss

            # [EPOCH FINISH]
            # Calculate total loss per epoch:
            disc_epoch_loss /= len(dataloader)
            gen_epoch_loss /= len(dataloader)
            disc_epoch_acc_real /= len(dataloader)
            disc_epoch_acc_fake /= len(dataloader)

            # Prepare some fake images for Tensorboard
            with torch.inference_mode():
                img, _ = next(iter(dataloader))
                noise = torch.randn(img.shape).to(device)
                img_fake = gen(noise)

            # Write to Tensorboard:
            imgs_real, _ = next(iter(dataloader))
            imgs_fake_grid = torchvision.utils.make_grid(img_fake.view(-1, 1, 28, 28),  # arg1.shape = [BATCH_SIZE, COLOR_CHANNELS, H, W] (See docstring)
                                                         nrow=12,
                                                         normalize=True)    
            imgs_real_grid = torchvision.utils.make_grid(imgs_real,  
                                                         nrow=12,
                                                         normalize=True)  
            # Model was loaded, update global_step for Tensorboad correctly
            global_step = json_log["epochs"] + epoch + 1    # Total epochs (Since model is loaded/saved)
            print(f"\n\nGlobal Step: {global_step}")
            writer.add_image("Fake Images", imgs_fake_grid, global_step=global_step)
            writer.add_image("Real Images", imgs_real_grid, global_step=global_step)
            writer.add_scalar("Generator Train Loss per epoch", gen_epoch_loss, global_step)
            writer.add_scalar("Discriminator Train Loss per epoch", disc_epoch_loss, global_step)
            writer.add_scalar("Discriminator Accuracy Real per epoch", disc_epoch_acc_real, global_step)
            writer.add_scalar("Discriminator Accuracy Fake per epoch", disc_epoch_acc_fake, global_step)

            # Write to JSON:
            text = f"Disc Loss: {disc_epoch_loss:.4f} Gen Loss: {gen_epoch_loss:.4f} Disc Acc Real: {disc_epoch_acc_real*100:.2f}% Disc Acc Fake: {disc_epoch_acc_fake*100:.2f}%"
            results.append(text)
            print(f"Epoch [{epoch+1}/{epochs}] {text}\n")

        # [TRAIN FINISH]
        # Calculate total train time:
        end_time = time.time()
        text = f"Training time: {helpers.format_seconds(end_time-start_time)}"
        print( f"\n{text}")

        # Write to JSON:
        json_log["results"] += results
        json_log["epochs"] = len(json_log["results"])
        json_log["train_durations"].append(f"Epochs: {epochs} {text}")
        helpers.write_json_log(filenames["dir"], filenames["gan"], json_log)

        # Save state_dict:
        helpers.save_or_load_model_checkpoint("save",
                                              filenames["dir"],
                                              filenames["generator"],
                                              gen, 
                                              gen_optim, 
                                              checkpoint={
                                                  "model_state_dict": gen.state_dict(),
                                                  "optimizer_state_dict": gen_optim.state_dict()
                                              })
        helpers.save_or_load_model_checkpoint("save",
                                              filenames["dir"],
                                              filenames["discriminator"],
                                              disc, 
                                              disc_optim, 
                                              checkpoint={
                                                  "model_state_dict": disc.state_dict(),
                                                  "optimizer_state_dict": disc_optim.state_dict()
                                              })

        # Tensorboard cleanup:
        writer.flush()
        writer.close()

    except KeyboardInterrupt:
        print("Keyboard Interrupt")


def main():
    # https://stackoverflow.com/questions/43763858/change-images-slider-step-in-tensorboard
    # tensorboard --samples_per_plugin "images=1000,scalars=5000" --logdir="./Prototypes/models state_dict/GAN_simple_v5/tensorboard"
    os.system("cls")

    version = 4
    filenames: IFilenames = {
        "dir": f"GAN_simple_v{version}",
        "generator": f"gen",
        "discriminator": f"disc",
        "gan": f"gan",
        "tensorboard": helpers.get_tensorboard_dir(f"GAN_simple_v{version}")
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_lr = 2e-4
    disc_lr = 1e-4
    batch_size = 32 * 4
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train, test, train_dataloader, test_dataloader = helpers.load_torch_dataset("MNIST", transform, batch_size)

    gen_0 = Generator(input_shape=28*28, hidden_units=256, output_shape=28*28)
    disc_0 = Discriminator(input_shape=28*28, hidden_units=256, output_shape=1)
    
    gen_0_optim = torch.optim.Adam(gen_0.parameters(), lr=gen_lr)
    disc_0_optim = torch.optim.Adam(disc_0.parameters(), lr=disc_lr)
    
    helpers.save_or_load_model_checkpoint("load", filenames["dir"], filenames["generator"], gen_0, gen_0_optim, device)
    helpers.save_or_load_model_checkpoint("load", filenames["dir"], filenames["discriminator"], disc_0, disc_0_optim, device)
    helpers.write_json_log(
        filenames["dir"],
        filenames["gan"],           
        {
            "batch_size": batch_size,
            "epochs": 0,
            "train_durations": [],
            "results": []
        },
        skip_if_exists=True
    )

    train_GAN(filenames=filenames,
              epochs=2, 
              device=device,
              dataloader=train_dataloader,
              gen=gen_0,
              gen_optim=gen_0_optim,
              disc=disc_0,
              disc_optim=disc_0_optim,
              criterion=nn.BCELoss(),
              skip=True)

    def view_result_images(gen: nn.Module,
                           disc: nn.Module,
                           rows: int,
                           cols: int):
        img, label = next(iter(train))  # img.shape = [1,28,28]
        plt.figure(figsize=(16, 9))
        gen.cpu()
        disc.cpu()
        gen.eval()
        disc.eval()
        with torch.inference_mode():
            for i in range(rows*cols):
                noise = torch.randn(img.shape)  # noise.shape = [1,28,28]
                noise = noise.unsqueeze(dim=0)  # noise.shape = [1,1,28,28]
                img_fake = gen(noise)
                is_fake_prob = disc(img_fake)
                is_fake_prob = is_fake_prob.item() * 100
                img_fake = img_fake.view(28, 28)  # img_fake.shape = [28,28]

                plt.subplot(rows, cols, i + 1)
                plt.imshow(img_fake, cmap="gray")
                plt.title(f"Prob to be real: {is_fake_prob:.2f}%", fontsize=10)
                plt.axis(False)
            plt.show()
    view_result_images(gen_0, disc_0, 5, 5)

    # summary(gen_0, (1, 28, 28))
    
    def export_onnx(gen: nn.Module):
        input = torch.randn([1,1,28,28]).to(device)
        gen.to(device)
        
        path = os.path.join(helpers.get_parent_dir(), env["MODELS_STATE_DICT_DIR"], filenames["dir"], f"gan.onnx")
        torch.onnx.export(gen,
                          input,
                          path,
                          input_names=["input"],
                          output_names=["output"],
                          dynamic_axes={
                              "input": { 0: "batch_size" },
                              "output": { 0: "batch_size" }
                          })
    export_onnx(gen_0)
    
    def playground():
        print()

    # playground()
main()

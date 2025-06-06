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


class IFilenames(TypedDict):
    dir: str
    generator: str
    discriminator: str
    gan: str
    tensorboard: str

# [MODEL] -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_channels: int, features: int, img_size: int):
        super().__init__()
        self.input_channels = input_channels
        self.features = features
        self.img_size = img_size
        # dilation=1;
        # Hout = (H + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
        self.disc = nn.Sequential(
            # [N, input_channels, 64, 64]
            self.conv2d_block(in_channels=input_channels,
                              out_channels=int(features/16),
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # H = W = (64 + 2 - 4)/2 + 1 = 31 + 1 = 32
            # [N, features/4, 32, 32]
            self.conv2d_block(in_channels=int(features/16),
                              out_channels=int(features/8),
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # H = W = (32 + 2 - 4)/2 + 1 = 15 + 1 = 16
            # [N, features/4, 16, 16]
            self.conv2d_block(in_channels=int(features/8),
                              out_channels=int(features/4),
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # H = W = (16 + 2 - 4)/2 + 1 = 7 + 1 = 8
            # [N, features/2, 8, 8]
            self.conv2d_block(in_channels=int(features/4),
                              out_channels=int(features/2),
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # H = W = (8 + 2 - 4)/2 + 1 = 3 + 1 = 4
            # [N, features/2, 4, 4]
            self.conv2d_block(in_channels=int(features/2),
                              out_channels=features,
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # [N, features, 2, 2]
            nn.Conv2d(in_channels=features,
                      out_channels=1,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            # [N, 1, 1, 1]
            nn.Flatten()
            # [N, 1]
        )

    def conv2d_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, out_channels: int, img_size: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_units = hidden_units
        self.out_channels = out_channels
        self.img_size = img_size

        self.gen = nn.Sequential(
            # [N, gen.out_channels, img_size, img_size] = [N, 1, 28, 28]
            nn.Flatten(),
            # [N, gen.out_channels * img_size * img_size] = [N, 28 * 28] = [N, 784]
            self.linear_block(in_features=in_features, out_features=hidden_units),
            # [N, hidden_units]
            self.linear_block(in_features=hidden_units, out_features=hidden_units),
            # [N, hidden_units]
            self.linear_block(in_features=hidden_units, out_features=hidden_units),
            # [N, hidden_units]
            nn.Linear(in_features=hidden_units, out_features=in_features),
            # [N, in_features] = [N, 784]
            nn.Tanh()            
        )

    def linear_block(self, in_features: int, out_features: int):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(num_features=out_features),
            nn.ReLU()
        )

    def forward(self, x):
        N = x.shape[0]
        return self.gen(x).view(N, self.out_channels, self.img_size, self.img_size) # [N, 3, 64, 64]
    
# Gradient Penalty    
def GP(disc: nn.Module, img_real: torch.Tensor, img_fake: torch.Tensor, device: str):
    N = img_real.shape[0]
    epsilon = torch.rand(N, 1, 1, 1, device=device, requires_grad=True).to(device)

    mixed_images = img_real*epsilon + img_fake*(1-epsilon)
    mixed_scores = disc(mixed_images)
    
    gradient = torch.autograd.grad(inputs=mixed_images,
                                   outputs=mixed_scores,
                                   grad_outputs=torch.ones_like(mixed_scores),
                                   create_graph=True,
                                   retain_graph=True)[0]
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty
    

# [TRAINING] -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train_GAN(filenames: IFilenames,
              epochs: int,
              device: str,
              dataloader_train,
              gen: nn.Module,
              gen_optim: torch.optim.Optimizer,
              disc: nn.Module,
              disc_optim: torch.optim.Optimizer,
              critic_iter: int,
              L: int,
              skip: bool = False,
              epochs_to_save_at: int = 500):
    if skip: return
    
    def save_model():
    # Save state_dict:
        helpers.save_or_load_model_checkpoint(
            "save",
            filenames["dir"],
            filenames["generator"],
            gen,
            gen_optim,
            checkpoint={
                "model_state_dict": gen.state_dict(),
                "optimizer_state_dict": gen_optim.state_dict()
            }
        )
        helpers.save_or_load_model_checkpoint(
            "save",
            filenames["dir"],
            filenames["discriminator"],
            disc,
            disc_optim,
            checkpoint={
                "model_state_dict": disc.state_dict(),
                "optimizer_state_dict": disc_optim.state_dict()
            }
        )
    
    # Tensorboard init:
    writer = SummaryWriter(log_dir=filenames["tensorboard"],
                           filename_suffix=Path(filenames["tensorboard"]).name)

    # JSON init:
    json_log = helpers.read_json_log(filenames["dir"], filenames["gan"])
    initial_global_step = json_log["epochs"]
    disc_epoch_loss, gen_epoch_loss = 0, 0

    # Train time start:
    start_time = time.time()
    true_start_time = time.time()

    try:
        # Models init:
        gen.to(device)
        disc.to(device)
        gen.train()
        disc.train()

        # Loop
        for epoch in tqdm(range(epochs)):
            disc_epoch_loss = 0
            gen_epoch_loss = 0
            for _, (img, _) in enumerate(dataloader_train):
                img = torch.as_tensor(img, device=device)

                disc_critic_loss = 0
                for i in range(critic_iter):
                    noise = torch.randn(img.shape).to(device)
                    img_fake = gen(noise)
                    
                    y_pred_fake = disc(img_fake)
                    y_pred_real = disc(img)
                    gp = GP(disc, img, img_fake, device)
                    disc_loss = torch.mean(y_pred_fake) - torch.mean(y_pred_real) + L * gp
                    disc_critic_loss += disc_loss
                    disc_optim.zero_grad()
                    disc_loss.backward(retain_graph=True)
                    disc_optim.step()
                    
                # [CRITIC FINISH]
                # Update generator weights:
                noise = torch.randn(img.shape).to(device)
                img_fake = gen(noise)
                y_pred_fake = disc(img_fake)
                gen_loss = -1 * torch.mean(y_pred_fake)
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()

                # Update results:
                disc_epoch_loss += (disc_critic_loss/critic_iter).item()
                gen_epoch_loss += gen_loss.item()

            # [EPOCH FINISH]
            # Calculate total loss per epoch:
            disc_epoch_loss /= len(dataloader_train)
            gen_epoch_loss /= len(dataloader_train)

            # Prepare some fake images for Tensorboard
            with torch.inference_mode():
                img, _ = next(iter(dataloader_train))
                noise = torch.randn(img.shape).to(device)
                img_fake = gen(noise)

            # Write to Tensorboard:
            imgs_real, _ = next(iter(dataloader_train))
            imgs_fake_grid = torchvision.utils.make_grid(img_fake,
                                                         nrow=8,
                                                         normalize=True)
            imgs_real_grid = torchvision.utils.make_grid(imgs_real,
                                                         nrow=8,
                                                         normalize=True)

            # Update global step (model is loaded/saved)
            global_step = initial_global_step + epoch + 1
            print(f"\n\nGlobal Step: {global_step}")
            writer.add_image("Fake images", imgs_fake_grid, global_step)
            writer.add_image("Real images", imgs_real_grid, global_step)
            writer.add_scalar("D LOSS/epoch", disc_epoch_loss, global_step)
            writer.add_scalar("G LOSS/epoch", gen_epoch_loss, global_step)

            # Logs:
            text = f"[D LOSS]: {disc_epoch_loss:.4f} [G LOSS]: {gen_epoch_loss:.4f}"
            json_log["results"].append(text)
            print(f"Epoch [{epoch+1}/{epochs}] {text}\n")
            
            # Save model every n epochs:
            if epoch > 0 and (epoch + 1) % epochs_to_save_at == 0:
                print(f"\nSaving model at epoch: {epoch + 1}")
                json_log["epochs"] = global_step
                end_time = time.time()
                train_time_text = f"[{device}] [Epochs: {epochs_to_save_at}] Training time: {helpers.format_seconds(end_time - start_time)}"
                print(f"{train_time_text}\n")
                json_log["train_durations"].append(train_time_text)
                helpers.write_json_log(filenames["dir"], filenames["gan"], json_log)
                save_model()
                start_time = time.time()

        # [TRAIN FINISH]
        # Save model and write to json:
        if epochs % epochs_to_save_at != 0:
            json_log["epochs"] = global_step
            end_time = time.time()
            train_time_text = f"[{device}] [Epochs: {epochs % epochs_to_save_at}] Training time: {helpers.format_seconds(end_time - start_time)}"
            print(f"\n{train_time_text}")
            json_log["train_durations"].append(train_time_text)
            helpers.write_json_log(filenames["dir"], filenames["gan"], json_log)
            save_model()
        
        print(f"\n[Epochs: {epochs}] Total train time: {helpers.format_seconds(end_time - true_start_time)}")

        # Tensorboard cleanup:
        writer.flush()
        writer.close()

    except KeyboardInterrupt:
        print("Keyboard Interrupt")

# [MAIN PROGRAM] -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # tensorboard --samples_per_plugin "images=1000,scalars=5000" --logdir="./Prototypes/models state_dict/WGP_DCGAN_Animal_Faces_v1/tensorboard"
    os.system("cls")

    version = 1
    filenames: IFilenames = {
        "dir": f"WGP_DCGAN_Animal_Faces_v{version}",
        "generator": f"gen",
        "discriminator": f"disc",
        "gan": f"gan",
        "tensorboard": helpers.get_tensorboard_dir(f"WGP_DCGAN_Animal_Faces_v{version}")
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_lr = 1e-4
    disc_lr = 1e-4
    batch_size = 32 * 2
    img_size = 64
    transform = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train, test, train_dataloader, test_dataloader = helpers.load_custom_img_dataset(
        "Animal faces",
        transform,
        batch_size,
        light=True
    )

    gen_0 = Generator(in_features=3*img_size*img_size, hidden_units=512, out_channels=3, img_size=img_size)
    disc_0 = Discriminator(input_channels=3, features=64 * 2**4, img_size=img_size)

    gen_0_optim = torch.optim.Adam(gen_0.parameters(), lr=gen_lr, betas=(0.0, 0.9))
    disc_0_optim = torch.optim.Adam(disc_0.parameters(), lr=disc_lr, betas=(0.0, 0.9))

    helpers.save_or_load_model_checkpoint("load", filenames["dir"], filenames["generator"], gen_0, gen_0_optim, device=device)
    helpers.save_or_load_model_checkpoint("load", filenames["dir"], filenames["discriminator"], disc_0, disc_0_optim, device=device)
    helpers.write_json_log(
        filenames["dir"],
        filenames["gan"],
        {
            "device": helpers.get_gpu_info("string"),
            "batch_size": batch_size,
            "epochs": 0,
            "train_durations": [],
            "results": []
        },
        skip_if_exists=True
    )
    
    train_GAN(filenames=filenames,
              epochs=100,
              device=device,
              dataloader_train=train_dataloader,
              gen=gen_0,
              gen_optim=gen_0_optim,
              disc=disc_0,
              disc_optim=disc_0_optim,
              critic_iter=5,
              L=10,
              skip=False,
              epochs_to_save_at=20)

    def view_result_images(gen: nn.Module,
                           disc: nn.Module,
                           N: int = 16,
                           nrow: int = 4):
        gen.to(device)
        disc.to(device)
        gen.eval()
        disc.eval()
        
        noise = torch.randn(N, gen.out_channels, gen.img_size, gen.img_size).to(device)
        with torch.inference_mode():
            imgs = gen(noise)
            imgs = torchvision.utils.make_grid(imgs, nrow=nrow, normalize=True)
            imgs_grid = torch.as_tensor(imgs).permute(1, 2, 0).detach().cpu().numpy()
                
        plt.figure(figsize=(16, 9))
        plt.suptitle("Generated Images")
        plt.imshow(imgs_grid)
        plt.axis(False)
        plt.show()
            
    view_result_images(gen_0, disc_0, 32, 8)
    
    def export_onnx(gen: nn.Module):
        input = torch.randn([1,100,1,1]).to(device)
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
    # export_onnx(gen_0)

    def test_gan(gen: nn.Module, disc: nn.Module):
        print("\n[TEST]\n")
        gen.to(device)
        disc.to(device)
        
        img, label = next(iter(train_dataloader))
        N = img.shape[0]
        
        noise = torch.randn(N, gen.out_channels, gen.img_size, gen.img_size).to(device)
        print(f"noise.shape: {noise.shape}")
        
        img_fake = gen(noise)
        print(f"img_fake.shape: {img_fake.shape}")

        img_fake_pred = disc(img_fake)
        print(f"img_fake_pred.shape: {img_fake_pred.shape}")
    
    test_gan(gen_0, disc_0)

main()

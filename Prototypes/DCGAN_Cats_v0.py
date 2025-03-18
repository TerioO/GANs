import math
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
    generator: str
    discriminator: str
    gan: str

# [MODEL] -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    **[MNIST DISCRIMINATOR]**

    Example:
    >>> y = gen(x)  # y.shape = [batch_size, 1, 28, 28]
    >>> disc = Discriminator(input_channels=1,  # Match MNIST channels
    >>>                      features=256)      # Should match generator features
    >>> y = disc(y) # y.shape = [batch_size, features]
    """

    def __init__(self, input_channels: int, features: int):
        super().__init__()
        self.input_channels = input_channels
        self.features = features
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
            nn.Conv2d(in_channels=int(features/2),
                      out_channels=features,
                      kernel_size=4,
                      stride=1,
                      padding=0),
            # H = W = (4 - 4)/1 + 1 = 1
            # [N, features, 1, 1]
            nn.Flatten(),
            # [N, features]
            nn.Sigmoid()
            # [N, features]
        )

    def conv2d_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding),
            # nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    """
    **[MNIST GENERATOR]**

    Every layer from the first doubles spatial resolution while halving the channels

    Example:
    >>> gen = Generator(input_channels=1,    # Noise channels (100 in the DCGAN paper)
    >>>                 features=256,        # Hyperparameter, should be a power of 2
    >>>                 output_channels=1)   # Match MNIST channels
    >>>
    >>> noise = torch.randn(batch_size, gen.input_channels, 1, 1)
    >>> y = gen(noise) # y.shape = [batch_size, 1, 28, 28]
    """

    def __init__(self, input_channels: int, features: int, output_channels: int):
        super().__init__()
        self.input_channels = input_channels
        self.features = features
        self.output_channels = output_channels
        # 4 16 32 64
        # dilation=1; output_padding=0 (defaults)
        # Hout = Wout = (H - 1) * stride - (2 * padding) + dilation * (kernel_size - 1) + output_padding + 1
        self.gen = nn.Sequential(
            # [N, input_channels, 1, 1]
            self.convTranspose2d_block(in_channels=input_channels,
                                       out_channels=int(features/2),
                                       kernel_size=4,
                                       stride=2,
                                       padding=0),
            # H = W = 0 - 0 + 3 + 1 = 4
            # [N, input_channels/2, 4, 4]
            self.convTranspose2d_block(in_channels=int(features/2),
                                       out_channels=int(features/4),
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
            # H = W = (4-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 6 - 2 + 3 + 1 = 6 + 2 = 8
            # [N, input_channels/4, 8, 8]
            self.convTranspose2d_block(in_channels=int(features/4),
                            out_channels=int(features/8),
                            kernel_size=4,
                            stride=2,
                            padding=1),
            # H = W = 14 - 2 + 4 = 16
            # [N, output_channels, 16, 16]
            self.convTranspose2d_block(in_channels=int(features/8),
                out_channels=int(features/16),
                kernel_size=4,
                stride=2,
                padding=1),
            # H = W = 30 - 2 + 4 = 32
            # [N, output_channels, 32, 32]
            nn.ConvTranspose2d(in_channels=int(features/16),
                               out_channels=output_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            # H = W = 62 - 2 + 4 = 64
            # [N, output_channels, 64, 64]
            nn.Tanh()
        )

    def convTranspose2d_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding),
            # nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.gen(x)
    
def init_weights(model: nn.Module):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(layer.weight.data, 0.0, 0.2)

# [TRAINING] -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train_GAN(filenames: IFilenames,
              epochs: int,
              device: str,
              dataloader_train,
              dataloader_test,
              gen: nn.Module,
              gen_optim: torch.optim.Optimizer,
              disc: nn.Module,
              disc_optim: torch.optim.Optimizer,
              criterion: nn.Module,
              skip: bool = False):
    if skip: return
    
    # Tensorboard init:
    writer = SummaryWriter(log_dir=os.path.join(helpers.get_tensorboard_dir(), filenames["gan"]),
                           filename_suffix=filenames["gan"])

    # JSON init:
    json_log = helpers.read_json_log(filenames["gan"])
    results = []
    disc_epoch_loss, gen_epoch_loss = 0, 0
    disc_epoch_acc_real, disc_epoch_acc_fake = 0, 0
    disc_epoch_acc_test = 0

    # Train time start:
    start_time = time.time()

    try:
        # Models init:
        gen.to(device)
        disc.to(device)
        gen.train()
        disc.train()

        # Loop
        for epoch in tqdm(range(epochs)):
            disc_epoch_acc_fake = 0
            disc_epoch_acc_real = 0
            disc_epoch_acc_test = 0
            disc_epoch_loss = 0
            gen_epoch_loss = 0
            for _, (img, _) in enumerate(dataloader_train):
                img = torch.as_tensor(img, device=device)

                # Generate a fake img from random noise:
                input_noise = torch.randn(img.shape[0], gen.input_channels, 1, 1).to(device)
                img_fake = gen(input_noise)

                # Update discriminator weights:
                y_pred_fake = disc(img_fake)
                y_pred_real = disc(img)
                disc_loss_fake = criterion(y_pred_fake, torch.zeros(y_pred_fake.shape).to(device))
                disc_loss_real = criterion(y_pred_real, torch.ones(y_pred_real.shape).to(device))
                disc_loss = (disc_loss_fake + disc_loss_real)/2
                disc_optim.zero_grad()
                disc_loss.backward(retain_graph=True)
                disc_optim.step()

                # Update generator weights:
                y_pred_fake = disc(img_fake)
                gen_loss = criterion(y_pred_fake, torch.ones(y_pred_fake.shape).to(device))
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()

                # Update results:
                disc_epoch_loss += disc_loss.item()
                gen_epoch_loss += gen_loss.item()
                disc_epoch_acc_fake += y_pred_fake.mean().item()
                disc_epoch_acc_real += y_pred_real.mean().item()

            # [EPOCH FINISH]
            # Calculate accuracy on test dataset:
            with torch.inference_mode():
                for _, (img, _) in enumerate(dataloader_test):
                    img = torch.as_tensor(img, device=device)
                    pred = disc(img)
                    pred = pred.mean().item()
                    disc_epoch_acc_test += pred
            
            # Calculate total loss per epoch:
            disc_epoch_loss /= len(dataloader_train)
            gen_epoch_loss /= len(dataloader_train)
            disc_epoch_acc_fake /= len(dataloader_train)
            disc_epoch_acc_real /= len(dataloader_train)
            disc_epoch_acc_test /= len(dataloader_test)

            # Prepare some fake images for Tensorboard
            with torch.inference_mode():
                img, _ = next(iter(dataloader_train))
                noise = torch.randn(img.shape[0], gen.input_channels, 1, 1).to(device)
                img_fake = gen(noise)

            # Write to Tensorboard:
            imgs_real, _ = next(iter(dataloader_train))
            imgs_fake_grid = torchvision.utils.make_grid(img_fake.view(-1, gen.output_channels, 64, 64),
                                                         nrow=11,
                                                         normalize=True)
            imgs_real_grid = torchvision.utils.make_grid(imgs_real,
                                                         nrow=11,
                                                         normalize=True)

            # Update global step (model is loaded/saved)
            global_step = json_log["epochs"] + epoch + 1
            print(f"\n\nGlobal Step: {global_step}")
            writer.add_image("Fake images", imgs_fake_grid, global_step)
            writer.add_image("Real images", imgs_real_grid, global_step)
            writer.add_scalar("D Acc REAL/epoch", disc_epoch_acc_real, global_step)
            writer.add_scalar("D Acc FAKE/epoch", disc_epoch_acc_fake, global_step)
            writer.add_scalar("D Acc REAL/epoch - TEST dataset", disc_epoch_acc_test, global_step)
            writer.add_scalar("D LOSS/epoch", disc_epoch_loss, global_step)
            writer.add_scalar("G LOSS/epoch", gen_epoch_loss, global_step)

            # Write to JSON:
            text = f"[D LOSS]: {disc_epoch_loss:.4f} [G LOSS]: {gen_epoch_loss:.4f} [D Acc REAL]: {disc_epoch_acc_real*100:.2f}% [D Acc FAKE]: {disc_epoch_acc_fake*100:.2f}% [D Acc REAL - TEST]: {disc_epoch_acc_test*100:.2f}%"
            results.append(text)
            print(f"Epoch [{epoch+1}/{epochs}] {text}\n")

        # [TRAIN FINISH]
        # Calculate train time:
        end_time = time.time()
        train_time_text = f"Training time: {helpers.format_seconds(end_time - start_time)}"
        print(f"\n{train_time_text}")

        # Write to JSON:
        json_log["results"] += results
        json_log["epochs"] = len(json_log["results"])
        json_log["train_durations"].append(f"[{device}] Epochs: {epochs} {train_time_text}")
        helpers.write_json_log(filenames["gan"], json_log)

        # Save state_dict:
        helpers.save_or_load_model_checkpoint("save",
                                              filenames["generator"],
                                              gen,
                                              gen_optim,
                                              checkpoint={
                                                  "model_state_dict": gen.state_dict(),
                                                  "optimizer_state_dict": gen_optim.state_dict()
                                              })
        helpers.save_or_load_model_checkpoint("save",
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

# [MAIN PROGRAM] -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    # tensorboard --samples_per_plugin "images=200,scalars=1000" --logdir="./Prototypes/tensorboard/DCGAN_Cats_v0_gan_0"
    os.system("cls")

    version = 1
    filenames: IFilenames = {
        "generator": f"DCGAN_Cats_v0_gen_{version}",
        "discriminator": f"DCGAN_Cats_v0_disc_{version}",
        "gan": f"DCGAN_Cats_v0_gan_{version}"
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_lr = 2e-4
    disc_lr = 1e-4
    batch_size = 32 * 3
    transform = transforms.Compose([
        transforms.Resize(size=(64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train, test, train_dataloader, test_dataloader = helpers.load_custom_img_dataset(
        "Cat and Dog",
        transform,
        batch_size,
        light=False
    )

    gen_0 = Generator(input_channels=100, features=64 * 2**4, output_channels=3)
    disc_0 = Discriminator(input_channels=3, features=64 * 2**4)

    gen_0_optim = torch.optim.Adam(gen_0.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    disc_0_optim = torch.optim.Adam(disc_0.parameters(), lr=disc_lr, betas=(0.5, 0.999))

    helpers.save_or_load_model_checkpoint("load", filenames["generator"], gen_0, gen_0_optim, device=device)
    helpers.save_or_load_model_checkpoint("load", filenames["discriminator"], disc_0, disc_0_optim, device=device)
    helpers.write_json_log(
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
              epochs=2,
              device=device,
              dataloader_train=train_dataloader,
              dataloader_test=test_dataloader,
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
        img, _ = next(iter(train))
        img = torch.as_tensor(img, device=device)
        plt.figure(figsize=(16, 9))
        plt.suptitle("Certainty that an image is real (90% --> REAL, 50% --> UNSURE, 10% --> FAKE)")
        gen.to(device)
        disc.to(device)
        gen.eval()
        disc.eval()
        with torch.inference_mode():
            for i in range(rows*cols):
                noise = torch.randn(1, gen.input_channels, 1, 1).to(device)
                img_fake = gen(noise)         # img_fake.shape = [1, gen.output_channels=3, 64, 64]
                certainty = disc(img_fake)    # certainty.shape = [1, disc.features=64*2^4]
                certainty = certainty.mean().item()
                
                # pytorch [N, C, H, W] --> imshow [H, W, C]
                # Remove batch dimension and rearrange remaining dimensions
                img_plt = img_fake.squeeze().permute(2, 1, 0).cpu().numpy()
                img_plt = np.clip(img_plt, 0, 1)
                plt.subplot(rows, cols, i+1)
                plt.imshow(img_plt)
                plt.title(f"{certainty*100:.2f}%")
                plt.axis(False)
            plt.show()
            
    view_result_images(gen_0, disc_0, 5, 5)


main()

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
    def __init__(self, input_channels: int, features: int, num_classes: int, img_size: int):
        super().__init__()
        self.input_channels = input_channels
        self.features = features
        self.num_classes = num_classes
        self.img_size = img_size
        
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)
        
        # dilation=1;
        # Hout = (H + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
        self.disc = nn.Sequential(
            # [N, input_channels, 128, 128]
            nn.Conv2d(in_channels=input_channels + num_classes,
                      out_channels=int(features/32),
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(0.2),
            # H = W = (128 + 2 - 4)/2 + 1 = 63 + 1 = 64
            # [N, features/32, 64, 64]
            self.conv2d_block(in_channels=int(features/32),
                              out_channels=int(features/16),
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # H = W = (64 + 2 - 4)/2 + 1 = 31 + 1 = 32
            # [N, features/16, 32, 32]
            self.conv2d_block(in_channels=int(features/16),
                              out_channels=int(features/8),
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # H = W = (32 + 2 - 4)/2 + 1 = 15 + 1 = 16
            # [N, features/8, 16, 16]
            self.conv2d_block(in_channels=int(features/8),
                              out_channels=int(features/4),
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # H = W = (16 + 2 - 4)/2 + 1 = 7 + 1 = 8
            # [N, features/4, 8, 8]
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
            nn.Linear(in_features=features, out_features=1),
            # [N, 1]
            nn.Sigmoid()
            # [N, 1]
        )

    def conv2d_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    # x.shape = [N, input_channels, 64, 64] 
    # labels.shape = [N]
    def forward(self, x, labels):
        embedding = self.embedding(labels)              # [labels.shape, num_classes] = [N, 3]
        embedding = embedding.unsqueeze(2).unsqueeze(3) # [N, 3, 1, 1]
        embedding = embedding.expand(-1, -1, self.img_size, self.img_size) # [N, 3, H, W]
        input = torch.cat([x, embedding], 1)            # [N, input_channels + 3, H, W]
        output = self.disc(input)                       # [N, 1]
        return output


class Generator(nn.Module):
    def __init__(self, input_channels: int, features: int, output_channels: int, num_classes: int, img_size: int):
        super().__init__()
        self.input_channels = input_channels
        self.features = features
        self.output_channels = output_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=num_classes)
        
        # dilation=1; output_padding=0 (defaults)
        # Hout = Wout = (H - 1) * stride - (2 * padding) + dilation * (kernel_size - 1) + output_padding + 1
        self.gen = nn.Sequential(
            # [N, input_channels, 1, 1]
            self.convTranspose2d_block(in_channels=input_channels + num_classes,
                                       out_channels=int(features/2),
                                       kernel_size=4,
                                       stride=2,
                                       padding=0),
            # H = W = 0 - 0 + 3 + 1 = 4
            # [N, features/2, 4, 4]
            self.convTranspose2d_block(in_channels=int(features/2),
                                       out_channels=int(features/4),
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
            # H = W = (4-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 6 - 2 + 3 + 1 = 6 + 2 = 8
            # [N, features/4, 8, 8]
            self.convTranspose2d_block(in_channels=int(features/4),
                                       out_channels=int(features/8),
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
            # H = W = 14 - 2 + 4 = 16
            # [N, features/8, 16, 16]
            self.convTranspose2d_block(in_channels=int(features/8),
                                       out_channels=int(features/16),
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
            # H = W = 30 - 2 + 4 = 32
            # [N, features/16, 32, 32]
            self.convTranspose2d_block(in_channels=int(features/16),
                                       out_channels=int(features/32),
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
            # H = W = 62 - 2 + 4 = 64
            # [N, features/32, 64, 64]
            nn.ConvTranspose2d(in_channels=int(features/32),
                               out_channels=output_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            # H = W = 126 - 2 + 4 = 128
            # [N, output_channels, 128, 128]
            nn.Tanh()
        )

    def convTranspose2d_block(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    # x.shape = [N, input_channels, 1, 1] 
    # labels.shape = [N]
    def forward(self, x, labels):
        embedding = self.embedding(labels)              # [labels.shape, num_classes] = [N, 3]
        embedding = embedding.unsqueeze(2).unsqueeze(3) # [N, 3, 1, 1]
        input = torch.cat([x, embedding], dim=1)        # [N, input_channels + 3, H, W]
        output = self.gen(input)                        # [N, 3, H, W] 
        return output 
    
# [TRAINING] -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def train_GAN(filenames: IFilenames,
              epochs: int,
              device: str,
              dataloader_train,
              gen: nn.Module,
              gen_optim: torch.optim.Optimizer,
              disc: nn.Module,
              disc_optim: torch.optim.Optimizer,
              criterion: nn.Module,
              skip: bool = False,
              epochs_to_save_at: int = 500):
    if skip: return
    
    def save_model():
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
    disc_epoch_acc_real, disc_epoch_acc_fake = 0, 0

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
            disc_epoch_acc_fake = 0
            disc_epoch_acc_real = 0
            disc_epoch_loss = 0
            gen_epoch_loss = 0
            
            for _, (img, labels) in enumerate(dataloader_train):
                img = torch.as_tensor(img, device=device)
                labels_real = torch.IntTensor(labels.type(torch.int)).to(device)   # [N]

                # Generate a fake img from random noise:
                input_noise = torch.randn(img.shape[0], gen.input_channels, 1, 1).to(device)               # [N, 100, 1, 1] 
                labels_fake = torch.randint(0, gen.num_classes, labels.shape, dtype=torch.int)
                labels_fake = torch.IntTensor(labels_fake).to(device) # [N]
                img_fake = gen(input_noise, labels_fake)

                # Update discriminator weights:
                y_pred_fake = disc(img_fake, labels_fake)
                y_pred_real = disc(img, labels_real)    
                disc_loss_fake = criterion(y_pred_fake, torch.zeros(y_pred_fake.shape).to(device))
                disc_loss_real = criterion(y_pred_real, torch.ones(y_pred_real.shape).to(device))
                disc_loss = (disc_loss_fake + disc_loss_real)/2
                disc_optim.zero_grad()
                disc_loss.backward(retain_graph=True)
                disc_optim.step()

                # Update generator weights:
                y_pred_fake = disc(img_fake, labels_fake)
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
            # Calculate total loss per epoch:
            disc_epoch_loss /= len(dataloader_train)
            gen_epoch_loss /= len(dataloader_train)
            disc_epoch_acc_fake /= len(dataloader_train)
            disc_epoch_acc_real /= len(dataloader_train)

            # Prepare some fake images for Tensorboard
            with torch.inference_mode():
                img, _ = next(iter(dataloader_train))
                N = img.shape[0]
                noise = torch.randn(img.shape[0], gen.input_channels, 1, 1).to(device)
                labels_grid = torch.arange(0, gen.num_classes, dtype=torch.int).repeat(math.ceil(N/gen.num_classes))
                labels_grid = torch.IntTensor(labels_grid[:N]).to(device) 
                img_fake = gen(noise, labels_grid)

            # Write to Tensorboard:
            imgs_real = helpers.make_grid_with_labels_in_order(N, dataloader_train, gen.num_classes)
            if imgs_real == None: imgs_real, _ = next(iter(dataloader_train))
            imgs_fake_grid = torchvision.utils.make_grid(img_fake,
                                                         nrow=9,
                                                         normalize=True)
            imgs_real_grid = torchvision.utils.make_grid(imgs_real,
                                                         nrow=9,
                                                         normalize=True)

            # Update global step (model is loaded/saved)
            global_step = initial_global_step + epoch + 1
            print(f"\n\nGlobal Step: {global_step}")
            writer.add_image("Fake images", imgs_fake_grid, global_step)
            writer.add_image("Real images", imgs_real_grid, global_step)
            writer.add_scalar("D Acc REAL/epoch", disc_epoch_acc_real, global_step)
            writer.add_scalar("D Acc FAKE/epoch", disc_epoch_acc_fake, global_step)
            writer.add_scalar("D LOSS/epoch", disc_epoch_loss, global_step)
            writer.add_scalar("G LOSS/epoch", gen_epoch_loss, global_step)

            # Logs:
            text = f"[D LOSS]: {disc_epoch_loss:.4f} [G LOSS]: {gen_epoch_loss:.4f} [D Acc REAL]: {disc_epoch_acc_real*100:.2f}% [D Acc FAKE]: {disc_epoch_acc_fake*100:.2f}%"
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
    # tensorboard --samples_per_plugin "images=1000,scalars=5000" --logdir="./Prototypes/models state_dict/CDCGAN_Animal_Faces_v5/tensorboard"
    os.system("cls")

    version = 5
    filenames: IFilenames = {
        "dir": f"CDCGAN_Animal_Faces_v{version}",
        "generator": f"gen",
        "discriminator": f"disc",
        "gan": f"gan",
        "tensorboard": helpers.get_tensorboard_dir(f"CDCGAN_Animal_Faces_v{version}")
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_lr = 2e-4
    disc_lr = 1e-4
    batch_size = 32
    img_size = 64 * 2
    transform = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train, test, train_dataloader, test_dataloader = helpers.load_custom_img_dataset(
        "Animal faces",
        transform,
        batch_size,
        light=False
    )

    gen_0 = Generator(input_channels=100, features=64 * 2**5, output_channels=3, num_classes=len(train.classes), img_size=img_size)
    disc_0 = Discriminator(input_channels=3, features=64 * 2**5, num_classes=len(train.classes), img_size=img_size)

    gen_0_optim = torch.optim.Adam(gen_0.parameters(), lr=gen_lr, betas=(0.5, 0.999))
    disc_0_optim = torch.optim.Adam(disc_0.parameters(), lr=disc_lr, betas=(0.5, 0.999))

    helpers.save_or_load_model_checkpoint("load", filenames["dir"], filenames["generator"], gen_0, gen_0_optim, device=device)
    helpers.save_or_load_model_checkpoint("load", filenames["dir"], filenames["discriminator"], disc_0, disc_0_optim, device=device)
    helpers.write_json_log(
        filenames["dir"],
        filenames["gan"],
        {
            "device": helpers.get_gpu_info("string"),
            "batch_size": batch_size,
            "epochs": 0,
            "gen_lr": gen_lr,
            "disc_lr": disc_lr,
            "train_durations": [],
            "results": [],
        },
        skip_if_exists=True
    )
    
    train_GAN(filenames=filenames,
              epochs=2,
              device=device,
              dataloader_train=train_dataloader,
              gen=gen_0,
              gen_optim=gen_0_optim,
              disc=disc_0,
              disc_optim=disc_0_optim,
              criterion=nn.BCELoss(),
              skip=True,
              epochs_to_save_at=40)

    def view_result_images(gen: nn.Module,
                           disc: nn.Module,
                           N: int = 16,
                           nrow: int = 4):
        gen.to(device)
        disc.to(device)
        gen.eval()
        disc.eval()
        
        imgs, labels = next(iter(train_dataloader))
        noise = torch.randn(N, gen.input_channels, 1, 1).to(device)
        labels = torch.arange(0, gen.num_classes, dtype=torch.int).repeat(math.ceil(N/gen.num_classes))
        labels = torch.IntTensor(labels[:N]).to(device) 
        with torch.inference_mode():
            imgs = gen(noise, labels)
            imgs = torchvision.utils.make_grid(imgs, nrow=nrow, normalize=True)
            imgs_grid = torch.as_tensor(imgs).permute(1, 2, 0).detach().cpu().numpy()
                
        plt.figure(figsize=(16, 9))
        plt.suptitle("Generated Images")
        plt.imshow(imgs_grid)
        plt.axis(False)
        plt.show()
            
    view_result_images(gen_0, disc_0, 180, 18)
    
    def export_onnx(gen: nn.Module):
        gen.to(device)
        
        noise = torch.randn([1,100,1,1]).to(device)
        label = torch.IntTensor(torch.tensor([0], dtype=torch.int)).to(device)
        
        path = os.path.join(helpers.get_parent_dir(), env["MODELS_STATE_DICT_DIR"], filenames["dir"], f"gan.onnx")
        torch.onnx.export(gen,
                          (noise, label),
                          path,
                          input_names=["noise", "labels"],
                          output_names=["output"],
                          dynamic_axes={
                              "noise": { 0: "batch_size" },
                              "labels": { 0: "batch_size" },
                              "output": { 0: "batch_size" }
                          })
    export_onnx(gen_0)

    def test_gan(gen: nn.Module, disc: nn.Module):
        print("\n[TEST]\n")
        gen.to(device)
        disc.to(device)
        
        imgs, labels = next(iter(train_dataloader))
        N = imgs.shape[0]
        noise = torch.randn(N, gen.input_channels, 1, 1).to(device)
        labels = torch.randint(0, gen.num_classes, labels.shape, dtype=torch.int)
        labels = torch.IntTensor(labels).to(device)
        
        img_fake = gen(noise, labels)
        print(f"img_fake.shape = ", img_fake.shape)
        
        pred = disc(img_fake, labels)
        print(f"pred.shape = {pred.shape}")

    test_gan(gen_0, disc_0)

main()

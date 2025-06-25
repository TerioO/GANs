import math
from pathlib import Path
import random
import time
import typing
import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
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
            # [N, input_channels, 28, 28]
            nn.Conv2d(in_channels=input_channels + num_classes,
                      out_channels=int(features/2),
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            # [N, features/2, 14, 14]
            self.conv2d_block(in_channels=int(features/2),
                              out_channels=features,
                              kernel_size=4,
                              stride=2,
                              padding=1),
            # [N, features, 7, 7]
            nn.Conv2d(in_channels=features,
                      out_channels=1,
                      kernel_size=7,
                      stride=1,
                      padding=0),
            # [N, 1, 1, 1]
            nn.Flatten(),
            # nn.Linear(in_features=features, out_features=1),
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

    # x.shape = [N, input_channels, 28, 28]
    # labels.shape = [N]
    def forward(self, x, labels):
        embedding = self.embedding(labels)              # [labels.shape, num_classes] = [N, 10]
        embedding = embedding.unsqueeze(2).unsqueeze(3) # [N, 10, 1, 1]
        embedding = embedding.expand(-1, -1, self.img_size, self.img_size)    # [N, 10, H, W]
        input = torch.cat([x, embedding], 1)            # [N, input_channels + 10, H, W]
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
                                       out_channels=features,
                                       kernel_size=7,
                                       stride=1,
                                       padding=0),
            # [N, features, 7, 7]
            self.convTranspose2d_block(in_channels=features,
                                       out_channels=int(features/2),
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
            # [N, features/2, 14, 14]
            nn.ConvTranspose2d(in_channels=int(features/2),
                               out_channels=output_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1),
            # [N, output_channels, 28, 28]
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
        embedding = self.embedding(labels)              # [labels.shape, num_classes] = [N, 10]
        embedding = embedding.unsqueeze(2).unsqueeze(3) # [N, 10, 1, 1]
        input = torch.cat([x, embedding], dim=1)        # [N, input_channels + 10, 1, 1]
        output = self.gen(input)                        # [N, 1, 28, 28] 
        return output                         
                         
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
    disc_epoch_acc_test = 0

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
            disc_epoch_acc_test = 0
            disc_epoch_loss = 0
            gen_epoch_loss = 0
            for _, (img, labels) in enumerate(dataloader_train):
                img = torch.as_tensor(img, device=device)
                labels = torch.IntTensor(labels.type(torch.int)).to(device)  # [N]

                # Generate a fake img from random noise:
                input_noise = torch.randn(img.shape[0], gen.input_channels, 1, 1).to(device)   # [N, 100, 1, 1]                            
                img_fake = gen(input_noise, labels)

                # Update discriminator weights:
                y_pred_fake = disc(img_fake, labels)
                y_pred_real = disc(img, labels)
                disc_loss_fake = criterion(y_pred_fake, torch.zeros(y_pred_fake.shape).to(device))
                disc_loss_real = criterion(y_pred_real, torch.ones(y_pred_real.shape).to(device))
                disc_loss = (disc_loss_fake + disc_loss_real)/2
                disc_optim.zero_grad()
                disc_loss.backward(retain_graph=True)
                disc_optim.step()

                # Update generator weights:
                y_pred_fake = disc(img_fake, labels)
                gen_loss = criterion(y_pred_fake, torch.ones(y_pred_fake.shape).to(device))
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()

                # Update loss tracking:
                disc_epoch_loss += disc_loss.item()
                gen_epoch_loss += gen_loss.item()
                # Update accuracy tracking:
                disc_epoch_acc_fake += y_pred_fake.mean().item()
                disc_epoch_acc_real += y_pred_real.mean().item()

            # [EPOCH FINISH]
            # Calculate accuracy on test dataset:
            with torch.inference_mode():
                for _, (img, labels) in enumerate(dataloader_test):
                    img = torch.as_tensor(img, device=device)
                    labels = torch.IntTensor(labels.type(torch.int)).to(device)
                    pred = disc(img, labels)
                    pred = pred.mean().item()
                    disc_epoch_acc_test += pred
            
            # Calculate total loss per epoch:
            disc_epoch_loss /= len(dataloader_train)
            gen_epoch_loss /= len(dataloader_train)
            # Calculate total accuracy per epoch:
            disc_epoch_acc_fake /= len(dataloader_train)
            disc_epoch_acc_real /= len(dataloader_train)
            disc_epoch_acc_test /= len(dataloader_test)

            # Prepare some fake and real images for Tensorboard
            with torch.inference_mode():
                img, _ = next(iter(dataloader_train))
                N = img.shape[0]
                noise = torch.randn(N, gen.input_channels, 1, 1).to(device)
                labels_grid = torch.arange(0, gen.num_classes, dtype=torch.int).repeat(math.ceil(N/gen.num_classes))
                labels_grid = torch.IntTensor(labels_grid[:N]).to(device)
                img_fake = gen(noise, labels_grid)
                
            imgs_real = helpers.make_grid_with_labels_in_order(N, dataloader_train, gen.num_classes)
            if imgs_real == None: imgs_real, _ = next(iter(dataloader_train))
            imgs_fake_grid = torchvision.utils.make_grid(img_fake,
                                                         nrow=gen.num_classes,
                                                         normalize=True)
            imgs_real_grid = torchvision.utils.make_grid(imgs_real,
                                                         nrow=gen.num_classes,
                                                         normalize=True)

            # Update global step (model is loaded/saved)
            global_step = initial_global_step + epoch + 1
            print(f"\n\nGlobal Step: {global_step}")
            
            # Calculate metric score every 5 epochs:
            if global_step % 5 == 0:
                metric_type = "KID"
                print(f"Computing {metric_type}...")
                score = helpers.metric_eval(gen, dataloader_train, device, 10, metric_type) # 640 real & 640 fake image
                print(f"{metric_type}: {score:.4f} \n")
                json_log["metric"].append(f"[Epoch: {global_step}] {metric_type}: {score:.4f}")
                writer.add_scalar(f"{metric_type}", score, global_step)
            
            # Write to tensorboard:
            writer.add_image("Fake images", imgs_fake_grid, global_step)
            writer.add_image("Real images", imgs_real_grid, global_step)
            writer.add_scalar("D Acc REAL/epoch", disc_epoch_acc_real, global_step)
            writer.add_scalar("D Acc FAKE/epoch", disc_epoch_acc_fake, global_step)
            writer.add_scalar("D Acc REAL/epoch - TEST dataset", disc_epoch_acc_test, global_step)
            writer.add_scalar("D LOSS/epoch", disc_epoch_loss, global_step)
            writer.add_scalar("G LOSS/epoch", gen_epoch_loss, global_step)

            # Logs:
            text = f"[D LOSS]: {disc_epoch_loss:.4f} [G LOSS]: {gen_epoch_loss:.4f} [D Acc REAL]: {disc_epoch_acc_real*100:.2f}% [D Acc FAKE]: {disc_epoch_acc_fake*100:.2f}% [D Acc REAL - TEST]: {disc_epoch_acc_test*100:.2f}%"
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
    # tensorboard --samples_per_plugin "images=1000,scalars=5000" --logdir="./Models/models state_dict/CDCGAN_FashionMNIST_v2/tensorboard"
    os.system("cls")

    version = 2
    filenames: IFilenames = {
        "dir": f"CDCGAN_FashionMNIST_v{version}",
        "generator": f"gen",
        "discriminator": f"disc",
        "gan": f"gan",
        "tensorboard": helpers.get_tensorboard_dir(f"CDCGAN_FashionMNIST_v{version}")
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen_lr = 2e-4
    disc_lr = 2e-4
    batch_size = 64
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train, test, train_dataloader, test_dataloader = helpers.load_torch_dataset("FashionMNIST",transform, batch_size)

    gen_0 = Generator(input_channels=100, features=512, output_channels=1, num_classes=len(train.classes), img_size=28)
    disc_0 = Discriminator(input_channels=1, features=512, num_classes=len(train.classes), img_size=28)

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
            "metric": [],
            "results": []
        },
        skip_if_exists=True
    )

    train_GAN(filenames=filenames,
              epochs=90,
              device=device,
              dataloader_train=train_dataloader,
              dataloader_test=test_dataloader,
              gen=gen_0,
              gen_optim=gen_0_optim,
              disc=disc_0,
              disc_optim=disc_0_optim,
              criterion=nn.BCELoss(),
              skip=True,
              epochs_to_save_at=500)

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

    view_result_images(gen_0, disc_0, 180, 20)

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
        print(f"img.shape: {imgs.shape}")
        print(f"labels.shape: {labels.shape}")
        
        N = batch_size
        noise = torch.randn(N, gen.input_channels, 1, 1).to(device)
        gen_labels = torch.randint(0, gen.num_classes, labels.shape, dtype=torch.int)
        gen_labels = torch.IntTensor(gen_labels).to(device)
        
        print(f"noise.shape: {noise.shape}")
        print(f"gen_labels.shape: {gen_labels.shape}")
        
        img_fake = gen(noise, gen_labels)
        print(f"img_fake.shape: {img_fake.shape}")   # [N, gen.output_channels=1, 28, 28]
        
        pred = disc(img_fake, gen_labels)   # [N, 1]
        print(f"pred.shape: {pred.shape}")
        
    # test_gan(gen_0, disc_0)
    
    def get_GAN_score():
        metric_type = "FID"
        print(f"Computing {metric_type}...")
        score = helpers.metric_eval(gen_0, train_dataloader, device, 20, metric_type)
        print(f"{metric_type}: {score}")
    
    # get_GAN_score()

main()

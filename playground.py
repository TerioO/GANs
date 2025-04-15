import torch
import torchvision
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
import json
import time
import helpers
import math
import subprocess


def linear(x: torch.Tensor, in_features: int, out_features: int):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    Condition: x.shape[1] == in_features

    :param x: Input tensor
    :param in_features: Size of each input sample
    :param out_features: Size of each output sample
    :return: tensorh.shape = [x.shape[0], out_features]
    """
    lin = nn.Linear(in_features=in_features, out_features=out_features)
    return lin(x)


def conv2d(x: torch.Tensor,
           in_channels: int,
           out_channels: int,
           kernel_size: int,
           stride: int = 1,
           padding: int = 0,
           dilation: int = 1,
           bias: bool = True):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Input Shape: [N, C, H, W]

    Output Shape: [N, Cout, Hout, Wout]

    C = in_channels

    Cout = out_channels

    Hout = (H + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1

    Wout = (W + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
    """
    layer_conv2d = nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             bias=bias)

    H_out = (x.shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
    W_out = (x.shape[3] + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1

    y = layer_conv2d(x)
    print(f"conv2d expected output shape: [{x.shape[0]}, {y.shape[1]}, {H_out}, {W_out}] | actual shape: {y.shape}")
    return y


def convTranspose2d(x: torch.Tensor,
                    in_channels: int,
                    out_channels: int,
                    kernel_size: int,
                    stride: int = 1,
                    padding: int = 0,
                    output_padding: int = 0,
                    dilation: int = 1,
                    bias: bool = True):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html

    Input Shape: [N, C, H, W]

    Output Shape: [N, Cout, Hout, Wout]

    C = in_channels

    Cout = out_channels

    Hout = (H - 1) * stride - (2 * padding) + dilation * (kernel_size - 1) + output_padding + 1

    Wout = (W - 1) * stride - (2 * padding) + dilation * (kernel_size - 1) + output_padding + 1
    """
    layer_convTranspose2d = nn.ConvTranspose2d(in_channels,
                                               out_channels,
                                               kernel_size,
                                               stride,
                                               padding,
                                               output_padding,
                                               dilation=dilation,
                                               bias=bias)

    H_out = (x.shape[2] - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
    W_out = (x.shape[3] - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1

    y = layer_convTranspose2d(x)
    print(f"convTranspose2d expected output shape: [{x.shape[0]}, {y.shape[1]}, {H_out}, {W_out}] | actual shape: {y.shape}")
    return y


def embedding(x: torch.LongTensor, num_embeddings: int, embedding_dim: int):
    """
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

    If x.shape = [32] --> y.shape = [32, embedding_dim]

    If x.shape = [32, 1, 28, 28] --> y.shape = [32, 1, 28, 28, embedding_dim]

    :param x: Input tensor (MUST BE **LongTensor** or **IntTensor**)
    :param num_embeddings: Size of the dictionary to extract embeddings
    :param embedding_dim: Size of each embedding vector
    :return tensor.shape = [x.shape, embedding_dim]
    """
    emb = nn.Embedding(num_embeddings, embedding_dim)
    return emb(x)


def conv2d_example_0():
    train, _, _, _ = helpers.load_torch_dataset("MNIST", transforms.ToTensor(), 32)
    img, _ = next(iter(train))

    # img.shape = [1,28,28]
    y = conv2d(img.unsqueeze(dim=0), 1, 4, 3)   # y.shape = [1,4,26,26]
    y = conv2d(y, 4, 8, 3)                      # y.shape = [1,8,24,24]
    y = conv2d(y, 8, 16, 3)                     # y.shape = [1,16,22,22]
    y = conv2d(y, 16, 32, 3)                    # y.shape = [1,32,20,20]
    y = y.view(-1, 20, 20)                      # y.shape = [32,20,20]

    # From the original image [1,28,28] after a few convolutions
    # we get a shape of [32,20,20] which means there are 32 images
    # of HxW = 20x20
    # They are not really images, they are FEATURES that help the network
    # in recognizing patterns in images
    plt.figure(figsize=(16, 6))
    for i in range(16):  # Plot only the first 16 feature maps
        plt.subplot(4, 4, i + 1)
        plt.imshow(y[i].detach().numpy(), cmap='gray')
        plt.title(f"Feature Map {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def convTranspose2d_example_0():
    train, _, _, _ = helpers.load_torch_dataset("MNIST", transforms.ToTensor(), 32)
    img, _ = next(iter(train))

    y_conv = conv2d(torch.as_tensor(img).unsqueeze(dim=0), 1, 8, 1, 1)
    y_convTranspose = convTranspose2d(y_conv, 8, 1, 2, 1)

    plt.figure(figsize=(9, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img.view(28, 28), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("After conv2d, convTranspose2d")
    plt.imshow(y_convTranspose.view(y_convTranspose.shape[2], y_convTranspose.shape[3]).detach().numpy(), cmap="gray")
    plt.show()


def DCGAN_example_MNIST():
    sigmoid = nn.Sigmoid()
    flatten = nn.Flatten()
    
    train, _, _, _ = helpers.load_torch_dataset("MNIST", transforms.ToTensor(), 32)
    img, _ = next(iter(train))
    x = torch.as_tensor(img).unsqueeze(dim=0)

    y = conv2d(x, 1, int(1024/4), 4, 2, 1)
    y = conv2d(y, int(1024/4), int(1024/2), 4, 2, 1)
    y = conv2d(y, int(1024/2), 1024, 7, 1, 0)
    y = flatten(y)
    y = sigmoid(y)
    print(y.shape)

    noise = torch.randn(1, 1024, 1, 1)
    y = convTranspose2d(noise, 1024, int(1024/2), 7, 1, 0)
    y = convTranspose2d(y, int(1024/2), int(1024/4), 4, 2, 1)
    y = convTranspose2d(y, int(1024/4), 1, 4, 2, 1)


def create_light_dataset():
    ta, te, ta_dataloader, te_dataloader = helpers.load_custom_img_dataset(
        "Cat and Dog",
        transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor()
        ]),
        32,
        light=True,
        purge=True,
        percent_train=0.05,
        percent_test=0.05,
        labels_count=0
    )

    print("Training: ", ta.classes, len(ta))
    print("Test: ", te.classes, len(te))
    print("Train batches: ", ta_dataloader.batch_size, len(ta_dataloader))
    print("Test batches: ", te_dataloader.batch_size, len(te_dataloader))


def view_batch_images(x: torch.Tensor, nrow: int, normalize: bool = False):
    grid = torchvision.utils.make_grid(x, nrow, normalize=normalize)
    plt.figure(figsize=(16,9))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis(False)
    plt.title("Batch of images")
    plt.show()

def main():
    os.system("cls")
    
    # [DATASET] ---------------------------------------------------------------------------------------------------------------
    batch_size = 32
    # train, test, train_dataloader, test_dataloader = helpers.load_torch_dataset(
    #     "MNIST",
    #     transforms.ToTensor(),
    #     batch_size
    # )
    train, test, train_dataloader, test_dataloader = helpers.load_custom_img_dataset(
        "Animal faces",
        transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]),
        batch_size,
        light=False,
        purge=False,
        percent_test=0.4,
        percent_train=0.4,
    )
    # print(x)
    num_classes = len(train.classes)
    classes = train.classes
    classes_to_idx = train.class_to_idx
    print(f"\n[DATASET]\n")
    print(f"num_classes: {num_classes}")
    print(f"classes: {classes}")
    print(f"classes_to_idx: {classes_to_idx}")
    
    imgs, labels = next(iter(train_dataloader))
    view_batch_images(imgs, 8, True)

    flatten = nn.Flatten()
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()

    # conv2d_example_0()
    # convTranspose2d_example_0()
    # DCGAN_example_MNIST()

    # create_light_dataset()
    # img = img.permute(2,1,0)
    # print(img.shape)
    # plt.imshow(img)
    # plt.show()


main()

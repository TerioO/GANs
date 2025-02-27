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

class Discriminator(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=4,
                      stride=2,
                      padding=1)
        )

    def block(self, in_channels: int, out_channels: int, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(0.2)
        )
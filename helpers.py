import torch
from typing import Literal, TextIO
from typing import IO
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from env import env
from tqdm import tqdm
import os
import shutil
import json
import inspect
from env import env


def load_MNIST(transform, batch_size: int):
    """
    Downloads/loads MNIST dataset from **torchvision** in **Datasets** root dir

    :param transform: A torchvision that transforms the data
    :param batch_size: int
    :return: ``train``, ``test``, ``train_dataloader``, ``test_dataloader``
    """

    train = torchvision.datasets.MNIST(root=env["DATASET_DIR"],
                                       train=True,
                                       download=True,
                                       transform=transform)
    test = torchvision.datasets.MNIST(root=env["DATASET_DIR"],
                                      train=False,
                                      download=True,
                                      transform=transform)
    train_dataloader = DataLoader(train, batch_size, True)
    test_dataloader = DataLoader(test, batch_size, True)
    return train, test, train_dataloader, test_dataloader


def save_or_load_model(model: nn.Module,  filename: str, mode: Literal["save", "load"], with_print: bool = False):
    """
    https://pytorch.org/tutorials/beginner/saving_loading_models.html

    Utility function for loading or saving the ``state_dict`` of a model.

    :param model: The model you want to save/load
    :param filename: Name of the file which contains the state_dict
    :param mode: Wether to save/load the state dict
    :param with_print: Choose if you want to print messaged in the console
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"])

    if not os.path.exists(cwd):
        os.mkdir(cwd)
    path = os.path.join(cwd, filename)
    if mode == "save":
        torch.save(model.state_dict(), path)
        if with_print:
            print(f"Model state dict saved at location: {path}")
    elif mode == "load":
        if not os.path.exists(path):
            if with_print:
                print(
                    f"Model state dict couldn't be loaded, state dict doesn't exist at: {path}")
            return
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        if with_print:
            print(f"Model state dicst loaded from: {path}")


def read_json_log(filename: str):
    """
    https://docs.python.org/3/library/json.html

    :param filename: The name of the JSON file, E.g: data.json
    :return: The JSON object as a Python dictionary
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"])

    path = os.path.join(cwd, filename)
    if not os.path.exists(path):
        return None
    return json.load(open(path, "r"))


def write_json_log(filename: str, json_obj, skip_if_exists: bool = False):
    """
    https://docs.python.org/3/library/json.html

    :param filename: The name of the JSON file, E.g: **data.json**
    :param json_obj: The JSON-like object you want to write in the file
    :param skip_if_exists: Will only write if the .json file doesn't exist, otherwise function immediately retuns
    Use with this param set to True when first creating the file
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"])

    if not os.path.exists(cwd):
        os.mkdir(cwd)
    path = os.path.join(cwd, filename)
    if skip_if_exists and os.path.exists(path):
        return
    # noinspection PyTypeChecker
    json.dump(json_obj, open(path, "w"), indent=4)

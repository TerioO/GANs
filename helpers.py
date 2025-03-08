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
import time
from pathlib import Path
from collections.abc import Callable
from typing import TypedDict, Dict
import math
import platform
import subprocess
from env import env

class ModelCheckpoint(TypedDict):
    model_state_dict: any
    optimizer_state_dict: any


def load_MNIST(transform, batch_size: int):
    """
    Downloads/loads MNIST dataset from **torchvision** in **Datasets** root dir

    :param transform: A torchvision.transforms to transform the data
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


def save_or_load_model_checkpoint(mode: Literal["save", "load"], filename: str, model: nn.Module, optim: torch.optim.Optimizer, device: str = None, checkpoint: ModelCheckpoint = None,  with_print: bool = False):
    """
    https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

    Utility function for loading or saving the ``state_dict`` of a model.
    
    :param mode: Wether to save/load the state dict
    :param filename: Name of the file which contains the state_dict
    :param model: The model you want to save/load
    :param optim: The models optimizer
    :param device: Device (Needed when loading)
    :param checkpoint: A dictionary with state_dicts for model and optimizer (Needed when saving)
    :param with_print: Choose if you want to print messaged in the console
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"])

    if not os.path.exists(cwd):
        os.mkdir(cwd)
    path = os.path.join(cwd, filename)
    if mode == "save":
        torch.save(checkpoint, path)
        if with_print:
            print(f"Model state dict saved at location: {path}")
    elif mode == "load":
        if not os.path.exists(path):
            if with_print:
                print(
                    f"Model state dict couldn't be loaded, state dict doesn't exist at: {path}")
            return
        loaded_checkpoint: ModelCheckpoint = torch.load(path, weights_only=True)
        model.to(device)  # Prevents device mismatch for the optimizer when loading
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        optim.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        model.eval()
        if with_print:
            print(f"Model state dict loaded from: {path}")


def read_json_log(filename: str):
    """
    https://docs.python.org/3/library/json.html

    :param filename: The name of the JSON file, E.g: **data** (.json extension appended automatically)
    :return: The JSON object as a Python dictionary
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"])

    path = os.path.join(cwd, f"{filename}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, "r"))


def write_json_log(filename: str, json_obj, skip_if_exists: bool = False):
    """
    https://docs.python.org/3/library/json.html

    :param filename: The name of the JSON file, E.g: **data** (.json extension appended automatically)
    :param json_obj: The JSON-like object you want to write in the file
    :param skip_if_exists: Will only write if the .json file doesn't exist, otherwise function immediately retuns.
    Use with this param set to True when first creating the file so that successive runs of the program don't overwite the .json file
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"])

    if not os.path.exists(cwd):
        os.mkdir(cwd)
    path = os.path.join(cwd, f"{filename}.json")
    if skip_if_exists and os.path.exists(path):
        return
    # noinspection PyTypeChecker
    json.dump(json_obj, open(path, "w"), indent=4)
    
def format_seconds(seconds: int):
    """
    Formats seconds into HH-MM-SS
    
    :param seconds: int
    :return: The text: "{hours}h - {minutes}m - {seconds}s"
    """
    hours =  math.floor(seconds/3600)
    minutes = math.floor(seconds%3600/60)
    seconds = math.floor(seconds%3600%60)
    text = f"{hours}h - {minutes}min - {seconds}s"
    return text

def get_tensorboard_dir():
    """
    Returns the path to the **tensorboard** dir which is created relative
    to file execution
    
    In ./example/test.py will return ./example/tensorboard
    
    In ./test.py will return ./tensorboard
    
    The name of the tensorboard dir can be changed in **env.py**
    """
    cwd = Path(inspect.stack()[1].filename).parent
    return os.path.join(cwd, env["TENSORBOARD_DIR"])

def get_gpu_info(returnType: Literal["dict", "string"]):
    """
    Get some info about the gpu
    
    :param returnType: Choose if you want to get gpu info as string or dict
    :return: The gpu info in a dict IF **torch.cuda.is_available()**
    
    >>> gpu = helpers.get_gpu_info("dict")
    >>> gpu["name"] 
    >>> gpu["memory"]
    >>> gpu["compute_probability"]
    >>> # OR
    >>> gpu = helpers.get_gpu_info("string")
    >>> gpu = "GPU: ... | NAME: ... | COMPUTE: ..."
    """
    
    if not torch.cuda.is_available(): return
    gpu = {}
    gpu_props = torch.cuda.get_device_properties()
    gpu["name"] = gpu_props.name
    gpu["memory"] = gpu_props.total_memory
    gpu["compute_capability"] = torch.cuda.get_device_capability()

    if returnType == "dist": return gpu
    elif returnType == "string":
        return f"GPU: {gpu["name"]} | MEMORY: {gpu['memory']} | COMPUTE: {gpu['compute_capability']}"
    
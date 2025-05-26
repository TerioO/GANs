import torch
from typing import Literal, List, TextIO, Tuple
from typing import IO
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
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
import timeit
import random
import warnings

# Disable torchvision "IS" and "KID" warning: https://github.com/JaidedAI/EasyOCR/issues/766
warnings.filterwarnings("ignore", category=UserWarning)

class ModelCheckpoint(TypedDict):
    model_state_dict: any
    optimizer_state_dict: any


def load_torch_dataset(dataset: Literal["MNIST", "FashionMNIST"], transform, batch_size: int):
    """
    Downloads/loads a given dataset from **torchvision** in **Datasets** root dir

    :param dataset: The name of the dataset available in **torchvision.datasets**
    :param transform: A **torchvision.transforms** to transform the data
    :param batch_size: int
    :return: ``train``, ``test``, ``train_dataloader``, ``test_dataloader``
    """

    train, test = 0, 0
    if dataset == "MNIST":
        train = torchvision.datasets.MNIST(root=env["DATASET_DIR"],
                                           train=True,
                                           download=True,
                                           transform=transform)
        test = torchvision.datasets.MNIST(root=env["DATASET_DIR"],
                                          train=False,
                                          download=True,
                                          transform=transform)
    elif dataset == "FashionMNIST":
        train = torchvision.datasets.FashionMNIST(root=env["DATASET_DIR"],
                                    train=True,
                                    download=True,
                                    transform=transform)
        test = torchvision.datasets.FashionMNIST(root=env["DATASET_DIR"],
                                          train=False,
                                          download=True,
                                          transform=transform)

    train_dataloader = DataLoader(train, batch_size, True)
    test_dataloader = DataLoader(test, batch_size, True)
    return train, test, train_dataloader, test_dataloader


def load_custom_img_dataset(dataset: Literal["Cat and Dog", "food-101", "Animal faces", "Human faces 2 genders", "Human faces emotions", "Manga faces", "Simpsons faces"], 
                            transform, 
                            batch_size: int, 
                            light: bool = False, 
                            purge: bool = False,
                            labels_count: int = 0, 
                            percent_train: float = 0.1, 
                            percent_test: float = 0.1):
    """
    Load a custom image dataset using **torchvision.datasets.ImageFolder**
    
    You must create a dedicated folder to store these datasets, when running with **light=True** the dir that is checked for existence
    is **env["DATASET_LIGHT_DIR"]**
    
    Custom image dataset shape, where *label_name* are the classes/labels of the samples:
    - "./dataset_name"
    - "./dataset_name/train"
    - "./dataset_name/train/label_name"
    - "./dataset_name/train/label_name/img_1.png"
    - "./dataset_name/test"
    - "./dataset_name/test/label_name"
    - "./dataset_name/test/label_name/img_1.png"
    
    A **train** dir is REQUIRED but a **test** dir is not. In this case return values of `test` and `test_dataloader` are `None`
    
    The *light* dataset has the same shape as the original dataset, but root dir is named as: **"{original_dataset_name} light"**
    
    If you run this function with **light=True** it will create a *light* version of the original dataset.
    The *light* version is created ONCE if the directories for it don't exist, otherwise it will return the 
    **torchvision.datasets.ImageFolder** and **DataLoader**
    
    If you want to recreate the *light* dataset, use **purge=True** which deletes the current *light* dataset
    and recreates it with your new options
    
    :param dataset: Must match the name of the dir where the dataset exists
    :param transform: A **torchvision.transforms** to transform the data
    :param batch_size: int
    :param light: If you want to use a light version of the original dataset, otherwise will return original dataset
    :param purge: If you want to delete the light dataset and recreate the light dataset
    :param labels_count: How many labels from the original dataset you want to copy (`labels_count <= 0` --> copies all the labels)
    :param percent_train: *Value in (0,1] interval*; How many train samples as percentage to copy from original dataset
    :param percent_test: *Value in (0,1] interval*; How many test samples as percentage to copy from original dataset
    :return: `train`, `test`, `train_dataloader`, `test_dataloader`
    """
    
    paths = {
        "dataset_light_dir": os.path.join(env["DATASET_LIGHT_DIR"], dataset),
        "train_dir_light": os.path.join(env["DATASET_LIGHT_DIR"], dataset, "train"),
        "test_dir_light": os.path.join(env["DATASET_LIGHT_DIR"], dataset, "test"),
        "train_dir": os.path.join(env["DATASET_DIR"], dataset, "train"),
        "test_dir": os.path.join(env["DATASET_DIR"], dataset, "test")
    }
    if not os.path.exists(env["DATASET_DIR"]):
        print(f"Check that '{env["DATASET_DIR"]}' dir exists!")
        print("Exiting")
        return
    if not os.path.exists(paths["train_dir"]): 
        print(f"Check that '{paths["train_dir"]}' dir exists")
        print("Exiting")
        return
    
    def getDataloaders(train_dir: str, test_dir: str):
        train = torchvision.datasets.ImageFolder(root=train_dir, transform=transform)
        train_dataloader = DataLoader(train, batch_size, True)
        
        test, test_dataloader = None, None
        if os.path.exists(test_dir): 
            test = torchvision.datasets.ImageFolder(root=test_dir, transform=transform)
            test_dataloader = DataLoader(test, batch_size, True)
        
        return train, test, train_dataloader, test_dataloader
    
    if purge and os.path.exists(paths['dataset_light_dir']): 
        print(f"\nDeleting '{paths['dataset_light_dir']}'\n")
        shutil.rmtree(paths["dataset_light_dir"])
        
    if light:
        if not os.path.exists(env["DATASET_LIGHT_DIR"]):
            print(f"Check that '{env["DATASET_LIGHT_DIR"]}' dir exists!")
            print("Exiting")
            return
        
        should_create = True if not os.path.exists(paths["dataset_light_dir"]) else False
        should_create_train = True if not os.path.exists(paths["train_dir_light"]) and os.path.exists(paths["train_dir"]) else False
        should_create_test = True if not os.path.exists(paths["test_dir_light"]) and os.path.exists(paths["test_dir"]) else False
        
        if should_create: os.mkdir(paths["dataset_light_dir"])
        
        about = {
            "percent_train": percent_train,
            "percent_test": percent_test,
            "len_light_train": 0,
            "len_light_test": 0,
            "classes_light_train": 0,
            "classes_light_test": 0,
            "len_train": 0,
            "len_test": 0,
            "classes_train": 0,
            "classes_test": 0
        }
        
        seed = timeit.default_timer()
        
        # [LIGHT TRAIN]
        if should_create_train:
            train_labels = os.listdir(paths["train_dir"])
            about["classes_train"] = [len(train_labels), train_labels]
            k_labels = labels_count
            if labels_count > len(train_labels) or labels_count <= 0: k_labels = len(train_labels) 
            random.seed(seed)
            train_labels = random.sample(train_labels, k=k_labels)
            about["classes_light_train"] = [len(train_labels), train_labels]
            
            if not os.path.exists(paths["dataset_light_dir"]): os.mkdir(paths["dataset_light_dir"])
            os.mkdir(paths["train_dir_light"])
            
            for label in tqdm(train_labels):
                os.mkdir(os.path.join(paths["train_dir_light"], label))
                images = os.listdir(os.path.join(paths["train_dir"], label))
                k = int(len(images)*percent_train)
                about["len_light_train"] += k
                about["len_train"] += len(images)
                for img in random.sample(images, k=k):
                    shutil.copy(src=f"{paths['train_dir']}/{label}/{img}", dst=f"{paths['train_dir_light']}/{label}/{img}")
        
        # [LIGHT TEST]
        if should_create_test:
            test_labels = os.listdir(paths["test_dir"])
            about["classes_test"] = [len(test_labels), test_labels]
            k_labels = labels_count
            if labels_count > len(test_labels) or labels_count <= 0: k_labels = len(test_labels) 
            random.seed(seed)
            test_labels = random.sample(test_labels, k=k_labels)
            about["classes_light_test"] = [len(test_labels), test_labels]
            
            if not os.path.exists(paths["dataset_light_dir"]): os.mkdir(paths["dataset_light_dir"])
            os.mkdir(paths["test_dir_light"])
            
            for label in tqdm(test_labels):
                os.mkdir(os.path.join(paths["test_dir_light"], label))
                images = os.listdir(os.path.join(paths["test_dir"], label))
                k = int(len(images)*percent_test)
                about["len_light_test"] += k
                about["len_test"] += len(images)
                for img in random.sample(images, k=k):
                    shutil.copy(src=f"{paths['test_dir']}/{label}/{img}", dst=f"{paths['test_dir_light']}/{label}/{img}")        
        
        # Write about.json
        json_path = os.path.join(paths["dataset_light_dir"], "about.json")
        if purge: json.dump(about, open(json_path, "w"), indent=4)
        
        return getDataloaders(paths["train_dir_light"], paths["test_dir_light"])
        
    else:
        if not os.path.exists(paths["train_dir"]): 
            print(f"Failed to create train dataloader. [Dir: '{paths["train_dir"]}' doesn't exist!]")
            return

        return getDataloaders(paths["train_dir"], paths["test_dir"])


def save_or_load_model_checkpoint(mode: Literal["save", "load"], dir: str, filename: str, model: nn.Module, optim: torch.optim.Optimizer, device: str = None, checkpoint: ModelCheckpoint = None,  with_print: bool = False):
    """
    https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

    Utility function for loading or saving the ``state_dict`` of a model.

    :param mode: Wether to save/load the state dict
    :param dir: Name of the dir where to save/load files
    :param filename: Name of the file which contains the state_dict
    :param model: The model you want to save/load
    :param optim: The models optimizer
    :param device: Device (Needed when loading)
    :param checkpoint: A dictionary with state_dicts for model and optimizer (Needed when saving)
    :param with_print: Choose if you want to print messaged in the console
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"], dir)

    if not os.path.exists(cwd):
        os.makedirs(cwd, exist_ok=True)
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
        # Prevents device mismatch for the optimizer when loading
        model.to(device)
        model.load_state_dict(loaded_checkpoint["model_state_dict"])
        optim.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        model.eval()
        if with_print:
            print(f"Model state dict loaded from: {path}")


def read_json_log(dir: str, filename: str):
    """
    https://docs.python.org/3/library/json.html

    :param dir: Name of the dir where to read json from
    :param filename: The name of the JSON file, E.g: **data** (.json extension appended automatically)
    :return: The JSON object as a Python dictionary
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"], dir)

    path = os.path.join(cwd, f"{filename}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path, "r"))


def write_json_log(dir: str, filename: str, json_obj, skip_if_exists: bool = False):
    """
    https://docs.python.org/3/library/json.html

    :param dir: Name of the dir where to write json to
    :param filename: The name of the JSON file, E.g: **data** (.json extension appended automatically)
    :param json_obj: The JSON-like object you want to write in the file
    :param skip_if_exists: Will only write if the .json file doesn't exist, otherwise function immediately retuns.
    Use with this param set to True when first creating the file so that successive runs of the program don't overwite the .json file
    """

    cwd = os.path.dirname(os.path.abspath(inspect.stack()[1].filename))
    cwd = os.path.join(cwd, env["MODELS_STATE_DICT_DIR"], dir)

    if not os.path.exists(cwd):
        os.makedirs(cwd, exist_ok=True)
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
    hours = math.floor(seconds/3600)
    minutes = math.floor(seconds % 3600/60)
    seconds = math.floor(seconds % 3600 % 60)
    text = f"{hours}h - {minutes}min - {seconds}s"
    return text


def get_tensorboard_dir(dir: str):
    """
    Constructs the path for the tensorboard dir
    
    :param dir: The directory where **tensorboard** dir is created
    :return: Path to **tensorboard** dir
    
    >>> Example
    # It will build a path in this format:
    f"./{env["MODELS_STATE_DICT_DIR"]}/{dir}/{env["TENSORBOARD_DIR"]}"
    """
    cwd = Path(inspect.stack()[1].filename).parent
    path = os.path.join(cwd, 
                env["MODELS_STATE_DICT_DIR"], 
                dir,
                env["TENSORBOARD_DIR"])
    return path


def get_parent_dir():
    """
    Get the working dir 
    
    If called in ./Prototypes/models state_dict => "models state_dict"
    
    If called in ./Prototypes => "Prototypes"
    """
    return Path(inspect.stack()[1].filename).parent


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

    if returnType == "dist":
        return gpu
    elif returnType == "string":
        return f"GPU: {gpu["name"]} | MEMORY: {gpu['memory']} | COMPUTE: {gpu['compute_capability']}"
    
    
def make_grid_with_labels_in_order(size: int, dataloader: DataLoader, num_classes: int):
    """
    Returns an array of ordered tensors by labels from 0 -> len(labels)
    
    :param size: How many tensor to include in final output
    :param dataloader: PyTorch Dataloader
    :param num_classes: How many classes to include [0, train.classes]

    >>> y = make_grid_with_labels_in_order(32, dataloader, 3)
    >>> # y.shape = [32, C, H, W]
    """
    tensors_to_add = []
    count = 0
    idx_to_skip = set()
    
    for _, (imgs, labels) in enumerate(dataloader):
        idx_to_skip = set()
        for _ in range(len(labels)):
            for j in range(len(labels)):
                if labels[j] == count and j not in idx_to_skip:
                    idx_to_skip.add(j)
                    tensors_to_add.append(imgs[j].unsqueeze(0))
                    count = int((count + 1) % num_classes)
                    if len(tensors_to_add) >= size: 
                        return torch.cat(tensors_to_add, dim=0)
                    break
    return None

def change_optim_lr(optim: torch.optim.Optimizer, global_step: int, steps_to_change_at: List[int], new_lrs: List[float]):
    """
    Change the optimizer's learning rate at some given **global_step**.
    
    Call this function after performing **global_step** update.
    
    :param optim: The optimizer.
    :param global_step: The **global_step** of the training.
    :param steps_to_change_at: Array of **global_step** values to change learning rate at.
    :param new_lrs: Array of new learning rate values, must have same size as the other array.
    """
    for i in range(len(steps_to_change_at)):
        if global_step == steps_to_change_at[i]:
            for p in optim.param_groups:
                p["lr"] = new_lrs[i]
                
def metric_eval(gen: nn.Module, dataloader, device: str, num_batches: int, metric_type: Literal["FID, KID, IS"], init_gen=False, empty_cuda_cache=True):
    """
    Calculate FID, IS or KID score. Can return the images used to calculate the score, in which case the return value is a tuple. 
    See example bellow on how to use this function.
    
    :param gen: The generator network for a CDCGAN
    :param dataloader: The dataloader
    :param device: PyTorch device 
    :param num_batches: How many batches to include in the result
    :param metric_type: Which **torchvision** metric to use from given choices
    :param init_gen: Sets generator in evaluation mode and to device, use this if generator is used outside training loop
    :param empty_cuda_cache: Option to release unoccupied memory held by caching allocators. Useful when doing evaluations inside training loops
    
    >>> fid_score = eval_fid(gen, train_dataloader, device, 4, "FID")
    >>> # fid_score = float
    >>> kid_score = eval_fid(gen, train_dataloader, device, 4, "KID")
    >>> # kid_score = float
    """
    def convert_imgs_to_torchmetrics_format(imgs: torch.Tensor):
        imgs = (imgs + 1) / 2.0
        if imgs.shape[1] == 1: # If image is grayscale, it needs to be turned in RGB
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = nn.functional.interpolate(imgs, size=(299, 299), mode="bilinear")
        return imgs

    if init_gen:
        gen.to(device)
        gen.eval()
    
    metric = None
    if metric_type == "FID": metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    elif metric_type == "KID": metric = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
    elif metric_type == "IS": metric = InceptionScore(normalize=True).to(device)
    
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        if batch_idx > num_batches:
            break
        
        # Real images:
        imgs_real = torch.as_tensor(imgs, device=device)
        imgs_real = convert_imgs_to_torchmetrics_format(imgs_real)
        
        # Fake images:
        N = imgs.shape[0]
        noise = torch.randn(N, gen.input_channels, 1, 1).to(device)
        labels = torch.IntTensor(labels.type(torch.int)).to(device)
        with torch.inference_mode():
            imgs_fake = gen(noise, labels)
            imgs_fake = convert_imgs_to_torchmetrics_format(imgs_fake)
        
        # Metric updates:
        if metric_type == "IS": metric.update(imgs_fake)
        else:
            metric.update(imgs_real, real=True)
            metric.update(imgs_fake, real=False)
    # [FOR END]
    
    score = metric.compute()
    metric.reset()
    if empty_cuda_cache: torch.cuda.empty_cache() 
    return score.item() if metric_type == "FID" else score[0].item()
    
def get_GAN_labels(global_step: int, steps_to_change_at: List[int], ranges: List[Tuple[float, float]], shape: torch.Size, ones) -> torch.Tensor:
    """
    Return a tensor in a uniform distribution given by range values or a **torch.ones(shape)/torch.zeros(shape)** if **ranges=(1,1)/(0,0)**.
    
    :param global_step: Training loop **global_step**
    :param steps_to_change_at: Will apply correspoding ranges for 
    :param ranges: Array of range tuples for the uniform distributions
    :param shape: The shape of the output tensor (a **torch.Size** object)
    :param ones: If global_step > steps_to_change_at, return either a **torch.ones()/torch.zeros()** 
    
    >>> y1 = helpers.get_GAN_labels(global_step, [100, 200], [(1, 1), (0.5, 0.6)], (2), ones=True)
    >>> # y = tensor([1., 1.]) (global_step = [0, 99])
    >>> # y = tensor([0.55, 0.5]) (global_step = [100, 199]) 
    >>> # y = tensor([1., 1.]) (global_step = [200, inf])
    """
    for i in range(len(steps_to_change_at)):
        if global_step < steps_to_change_at[i]:
            r1, r2 = ranges[i][0], ranges[i][1]
            if r1 == r2 == 0: return torch.zeros(shape)
            if r1 == r2 == 1: return torch.ones(shape)
            else: return torch.FloatTensor(shape).uniform_(r1, r2)
    
    return torch.ones(shape) if ones else torch.zeros(shape)      
# Generative Adversarial Networks

This repository will contain some GAN applications.

## Project structure
- 📁 `Datasets` at root will contain all the datasets used for every model (not saved on git)
- 📁 `tensorboard` at any depth will contain data for **Tensorboard** (not saved on git)
- 📁 `models state_dict` dir will contain the models state_dict as well as a json file detailing training, model
- 📁 `Prototypes` will contain prototypes used for testing and tunning the models before releasing the final version
architecture, hyperparameters and some results (loss, accuracy, etc.)
- `env.py` contains state used across the project, currently stores the names for directories (Datasets, tensorboard, ...) which you can change
the name of here if you want
- `helpers.py` will contain reusable functions for the scripts (they all come with documentation, read and use them as needed)
- `playground.py` a place where you can test Pytorch layers usage, anything really


## Datasets that need to be downloaded manually
- Cats & Dogs

## Packages for this project

 - [PyTorch & torchvision](https://pytorch.org/get-started/locally/)
 - [Numpy](https://numpy.org/install/)
 - [Pandas](https://pandas.pydata.org/docs/getting_started/index.html)
 - [Matplotlib](https://matplotlib.org/stable/install/index.html)
 - [tqdm](https://pypi.org/project/tqdm/)
 - [Tensorboard for Pytorch](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html#run-tensorboard)
 - [PIL](https://pypi.org/project/pillow/)
 - [Torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
 - [torch-summary](https://pypi.org/project/torch-summary/)
 - [onnx](https://pypi.org/project/onnx/) You might need a previous version to make it work
 - [onnxruntime](https://pypi.org/project/onnxruntime/)
 - [protobuf](https://pypi.org/project/protobuf/) (For onnx - might not be needed)
 - [cmake](https://pypi.org/project/cmake/) (For onnx - might not be needed)
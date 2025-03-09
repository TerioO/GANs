import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

env = {
    # Name for the dir where all datasets are stored, used in helpers.py
    "DATASET_DIR": os.path.join(BASE_DIR, "Datasets"),
    
    # Name for the first where state_dicts are saved, used in helpers.py
    "MODELS_STATE_DICT_DIR": "models state_dict",
    
    # Name for the dir where tensorboard saves/reads data, used in helpers.py
    "TENSORBOARD_DIR": "tensorboard"
}

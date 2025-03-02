import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

env = {
    "DATASET_DIR": os.path.join(BASE_DIR, "Datasets"),
    "MODELS_STATE_DICT_DIR": "models state_dict",
    "TENSORBOARD_DIR": "tensorboard"
}

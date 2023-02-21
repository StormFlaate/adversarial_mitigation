


import torch
from config import RESNET18_MODEL_NAME, TRAIN_DATASET_ROOT_DIR
from misc_helper import save_model_and_parameters_to_file


model_inceptionv3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

save_model_and_parameters_to_file(model_inceptionv3, RESNET18_MODEL_NAME, TRAIN_DATASET_ROOT_DIR, models_dir="models")
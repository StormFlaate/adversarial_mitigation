# Importing the required libraries
from typing import Tuple
import torch # Import the Pytorch library
import torchvision # Import the torchvision library
from torchvision import datasets, transforms # Import the transforms module from torchvision


import numpy as np
from PIL import Image # Import the Image module from the Python Imaging Library (PIL)
import matplotlib.pyplot as plt

import urllib # Import the urllib library for URL handling
import sys
from tqdm import tqdm
from customDataset import ISICDataset

# HELPER FUNCTIONS
from data_exploration_helper import dataset_overview


#########################################################
# ==================== CONSTANTS ========================
#########################################################
# Training set 2018
TRAIN_2018_LABELS: str = "./data/ISIC2018_Training_GroundTruth.csv"
TRAIN_2018_ROOT_DIR: str = "./data/ISIC2018_Training_Input"

TEST_2018_LABELS: str = "./data/ISIC2018_Validation_GroundTruth.csv"
TEST_2018_ROOT_DIR: str = "./data/ISIC2018_Validation_Input"

# Dataset 2019 - has not been split into train and test
DATASET_2019_LABELS: str = "./data/ISIC_2019_Training_GroundTruth.csv"
DATASET_2019_ROOT_DIR: str = "./data/ISIC_2019_Training_Input"

BATCH_SIZE: int = 64
EPOCH_COUNT: int = 20

# NUMBER OF ROWS - can be used if you want to run some simple tests
TRAIN_NROWS: int = None # SET TO None if you want all samples
TEST_NROWS: int = None # SET TO None if you want all samples


LEARNING_RATE: float = 0.001
MOMENTUM: float = 0.9


IMAGENET_MEAN: Tuple = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple = (0.229, 0.224, 0.225)

INCEPTIONV3_PIXEL_SIZE: int = 299
RESNET18_PIXEL_SIZE: int = 224

IMAGE_FILE_TYPE: str = "jpg"

INCEPTIONV3_MODEL_NAME: str = "inceptionv3"
RESNET18_MODEL_NAME: str = "resnet18"
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

# ORIGINAL DATA
# Training set 2018
TRAIN_2018_LABELS: str = "./data/ISIC2018_Training_GroundTruth.csv"
TRAIN_2018_ROOT_DIR: str = "./data/ISIC2018_Training_Input"

TEST_2018_LABELS: str = "./data/ISIC2018_Validation_GroundTruth.csv"
TEST_2018_ROOT_DIR: str = "./data/ISIC2018_Validation_Input"

# Dataset 2019 - has not been split into train and test
DATASET_2019_LABELS: str = "./data/ISIC_2019_Training_GroundTruth.csv"
DATASET_2019_ROOT_DIR: str = "./data/ISIC_2019_Training_Input"

# AUGMENTED DATA
# Training set 2018
AUGMENTED_TRAIN_2018_LABELS: str = "./augmented_data/ISIC2018_Training_GroundTruth.csv"
AUGMENTED_TRAIN_2018_ROOT_DIR: str = "./augmented_data/ISIC2018_Training_Input"

AUGMENTED_TEST_2018_LABELS: str = "./augmented_data/ISIC2018_Validation_GroundTruth.csv"
AUGMENTED_TEST_2018_ROOT_DIR: str = "./augmented_data/ISIC2018_Validation_Input"

# Dataset 2019 - has not been split into train and test
AUGMENTED_DATASET_2019_LABELS: str = "./augmented_data/ISIC_2019_Training_GroundTruth.csv"
AUGMENTED_DATASET_2019_ROOT_DIR: str = "./augmented_data/ISIC_2019_Training_Input"


# which data to choose
TRAIN_DATASET_LABELS: str = AUGMENTED_TRAIN_2018_LABELS
TRAIN_DATASET_ROOT_DIR: str = AUGMENTED_TRAIN_2018_ROOT_DIR

TEST_DATASET_LABELS: str = TEST_2018_LABELS 
TEST_DATASET_ROOT_DIR: str = TEST_2018_ROOT_DIR


# NUMBER OF ROWS - can be used if you want to run some simple tests
TRAIN_NROWS: int = 10 # SET TO None if you want all samples
TEST_NROWS: int = 10 # SET TO None if you want all samples



# hypter parameters
IMAGENET_MEAN: Tuple = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple = (0.229, 0.224, 0.225)

BATCH_SIZE: int = 64
STEP_SIZE: int = 10
GAMMA: int = 0.1
EPOCH_COUNT: int = 30
LEARNING_RATE: float = 0.001
MOMENTUM: float = 0.9

# NEED TO PUT THE TEST AND TRAIN DATASET TOGETHER, and create a validation set with eq

# Data augmentation
MIN_NUMBER_OF_EACH_CLASS: int = 2000
RANDOM_HORIZONTAL_FLIP_PROBABILITY: float = 0.25
RANDOM_VERTICAL_FLIP_PROBABILITY: float = 0.25
RAND_AUGMENT_NUM_OPS: int = 4
RAND_AUGMENT_MAGNITUDE: int = 2
RAND_AUGMENT_NUM_MAGNITUDE_BINS: int = 10



IMAGE_FILE_TYPE: str = "jpg"

INCEPTIONV3_MODEL_NAME: str = "inceptionv3"
RESNET18_MODEL_NAME: str = "resnet18"

INCEPTIONV3_PIXEL_SIZE: int = 299
RESNET18_PIXEL_SIZE: int = 224

NUM_WORKERS: int = 10


# fastest for resnet18 batch size 256 -> num workers 10
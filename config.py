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

##################################################################
# ==================== DATASETS DEFINTION ========================
##################################################################

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


######################################
# ========== CHOOSE DATASET ==========
######################################
TRAIN_DATASET_LABELS: str = AUGMENTED_TRAIN_2018_LABELS
TRAIN_DATASET_ROOT_DIR: str = AUGMENTED_TRAIN_2018_ROOT_DIR

TEST_DATASET_LABELS: str = TEST_2018_LABELS 
TEST_DATASET_ROOT_DIR: str = TEST_2018_ROOT_DIR


# NUMBER OF ROWS - can be used if you want to run some simple tests
TRAIN_NROWS: int = None # SET TO None if you want all samples
TEST_NROWS: int = None # SET TO None if you want all samples



#########################################################
# ==================== PARAMETERS =======================
#########################################################

# PARAMETERS - Transforms
IMAGENET_MEAN: Tuple = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple = (0.229, 0.224, 0.225)
INCEPTIONV3_PIXEL_SIZE: int = 299
RESNET18_PIXEL_SIZE: int = 224

# PARAMETERS - Optimizer
LEARNING_RATE: float = 0.001
MOMENTUM: float = 0.9

# PARAMETERS - Scheduler
STEP_SIZE: int = 5
GAMMA: int = 0.1

# PARAMETERS - Dataloader
BATCH_SIZE: int = 64
VAL_BATCH_SIZE: int = 1
PIN_MEMORY_TRAIN_DATALOADER: bool = True
SHUFFLE_TRAIN_DATALOADER: bool = True
SHUFFLE_VAL_DATALOADER: bool = True

# PARAMETERS - Model Training
EPOCH_COUNT: int = 5
TRAIN_SPLIT_PERCENTAGE: float = 0.9
VAL_SPLIT_PERCENTAGE: float = 1 - TRAIN_SPLIT_PERCENTAGE


# PARAMETERS - DATA AUGMENTATION
MIN_NUMBER_OF_EACH_CLASS: int = 6000
RAND_AUGMENT_NUM_OPS: int = 4
RAND_AUGMENT_MAGNITUDE: int = 2
RAND_AUGMENT_NUM_MAGNITUDE_BINS: int = 10

RANDOM_VERTICAL_FLIP_PROBABILITY: float = 0.25
RANDOM_HORIZONTAL_FLIP_PROBABILITY: float = 0.25
MIN_MAX_ROTATION_RANGE: Tuple = (-90, 90)


# PARAMETERS - MISCELLANEOUS
IMAGE_FILE_TYPE: str = "jpg"
INCEPTIONV3_MODEL_NAME: str = "inceptionv3"
RESNET18_MODEL_NAME: str = "resnet18"

MODEL_NAME: str = RESNET18_MODEL_NAME


# PARAMETERS - GPU
NUM_WORKERS: int = 10


# TRANSFORMS FOR RESNET-18 and INCEPTION V3
# Define image pre-processing steps
PREPROCESS_RESNET18 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(RESNET18_PIXEL_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Define image pre-processing steps
PREPROCESS_INCEPTIONV3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(INCEPTIONV3_PIXEL_SIZE),
    transforms.CenterCrop(INCEPTIONV3_PIXEL_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

PREPROCESS_TRANSFORM = PREPROCESS_RESNET18



"""
Network training was performed using two NVIDIA GTX 1080Ti cards and the Caffe [9] framework.
 As optimizer, SGD was chosen with learning rate starting at 0.01, weight decay and momentum equal to 0.0001 and 0.9 respectively.
The maximum number of iterations has been set at 75000, decreasing the learning rate by a factor of 10 at each step of 20000 iterations. Finally, the 0.008 value was used for the 𝛾
 parameter in the Eq. 2. Regarding SVM classifier, an RBF kernel with 𝜆=0.01
 and 𝐶=10
 were used.
"""
# Importing the required libraries
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

BATCH_SIZE: int = 512
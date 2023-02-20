# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import torchvision
from torchvision import datasets, transforms

# Import custom modules
from config import (AUGMENTED_TEST_2018_LABELS, AUGMENTED_TEST_2018_ROOT_DIR, 
                    AUGMENTED_TRAIN_2018_LABELS, AUGMENTED_TRAIN_2018_ROOT_DIR, 
                    MIN_NUMBER_OF_EACH_CLASS, RAND_AUGMENT_MAGNITUDE, RAND_AUGMENT_NUM_MAGNITUDE_BINS, RAND_AUGMENT_NUM_OPS, RANDOM_HORIZONTAL_FLIP_PROBABILITY, 
                    RANDOM_VERTICAL_FLIP_PROBABILITY, TEST_2018_LABELS, TEST_2018_ROOT_DIR, 
                    TRAIN_2018_LABELS, TRAIN_2018_ROOT_DIR)
from customDataset import ISICDataset
from data_exploration_helper import dataset_overview
from data_augmentation import augment_images_and_save_to_file



# Define image pre-processing steps
# Define the transforms to apply to the training data
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=RANDOM_HORIZONTAL_FLIP_PROBABILITY),
    transforms.RandomVerticalFlip(p=RANDOM_VERTICAL_FLIP_PROBABILITY),
    transforms.RandAugment(num_ops=RAND_AUGMENT_NUM_OPS, magnitude=RAND_AUGMENT_MAGNITUDE, num_magnitude_bins=RAND_AUGMENT_NUM_MAGNITUDE_BINS),
    transforms.ToPILImage()
])




augment_images_and_save_to_file(TRAIN_2018_ROOT_DIR, AUGMENTED_TRAIN_2018_ROOT_DIR, TRAIN_2018_LABELS, AUGMENTED_TRAIN_2018_LABELS, augmentation_transform, min_number_of_each_class=MIN_NUMBER_OF_EACH_CLASS)
augment_images_and_save_to_file(TEST_2018_ROOT_DIR, AUGMENTED_TEST_2018_ROOT_DIR, TEST_2018_LABELS, AUGMENTED_TEST_2018_LABELS, augmentation_transform, min_number_of_each_class=MIN_NUMBER_OF_EACH_CLASS)



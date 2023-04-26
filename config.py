# Importing the required libraries
from typing import Tuple
from torchvision import transforms # Import the transforms module from torchvision

#########################################################
# ==================== PARAMETERS =======================
#########################################################

# PARAMETERS - Transforms
INCEPTIONV3_PIXEL_SIZE: int = 299
RESNET18_PIXEL_SIZE: int = 224

# PARAMETERS - Optimizer
LEARNING_RATE: float = 0.01
MOMENTUM: float = 0.9

# PARAMETERS - Scheduler
STEP_SIZE: int = 10
GAMMA: int = 0.1

# PARAMETERS - Dataloader
BATCH_SIZE: int = 1
PIN_MEMORY_TRAIN_DATALOADER: bool = True
SHUFFLE_TRAIN_DATALOADER: bool = True
SHUFFLE_VAL_DATALOADER: bool = True

# PARAMETERS - Model Training
# the epoch used for training the model
EPOCH_COUNT: int = 50

# 2018: 0.8, 0.2, test_dataset
TRAIN_SPLIT_2018: float = 0.8
VAL_SPLIT_2018: float = 0.2

# 2019: 0.7, 0.2, 0.1
TRAIN_SPLIT_2019: float = 0.7
VAL_SPLIT_2019: float = 0.2
TEST_SPLIT_2019: float = 1 - TRAIN_SPLIT_2019 - VAL_SPLIT_2019

# NUMBER OF ROWS - can be used if you want to run some simple tests
TRAIN_NROWS: int = None # SET TO None if you want all samples
TEST_NROWS: int = None # SET TO None if you want all samples




#########################################################################
# ==================== DO NOT CHANGE - PARAMETERS =======================
#########################################################################
# PARAMETERS - DATA AUGMENTATION
MIN_NUMBER_OF_EACH_CLASS_2018: int = 3000
MIN_NUMBER_OF_EACH_CLASS_2019: int = 6000
RANDOM_VERTICAL_FLIP_PROBABILITY: float = 0.25
RANDOM_HORIZONTAL_FLIP_PROBABILITY: float = 0.25
MIN_MAX_ROTATION_RANGE: Tuple = (-90, 90)

# PARAMETERS - MISCELLANEOUS
IMAGE_FILE_TYPE: str = "jpg"
INCEPTIONV3_MODEL_NAME: str = "inception_v3"
RESNET18_MODEL_NAME: str = "resnet18"
RANDOM_SEED: int = 42

# PARAMETERS - GPU
NUM_WORKERS: int = 1

# TRANSFORMS FOR RESNET-18 and INCEPTION V3
# Define image pre-processing steps
PREPROCESS_RESNET18 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(RESNET18_PIXEL_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Define image pre-processing steps
PREPROCESS_INCEPTIONV3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(INCEPTIONV3_PIXEL_SIZE),
    transforms.CenterCrop(INCEPTIONV3_PIXEL_SIZE),
    transforms.ToTensor(),
    # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


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

# Dataset 2019 - has not been split into train and test
AUGMENTED_DATASET_2019_LABELS: str = (
    "./augmented_data/ISIC_2019_Training_GroundTruth.csv"
)
AUGMENTED_DATASET_2019_ROOT_DIR: str = "./augmented_data/ISIC_2019_Training_Input"



##################################################################
# ==================== TRAINED MODELS ===========================
##################################################################

TRAINED_INCEPTION_V3_MODEL_2019: str = (
    "inception_v3_augmented_data_ISIC_2019_Training_Input_2023-04-24_50__78e.pt"
)
TRAINED_RESNET18_MODEL_2019: str = (
    "resnet18_augmented_data_ISIC_2019_Training_Input_2023-04-24_50__c6c.pt"
)
TRAINED_INCEPTION_V3_MODEL_2018: str = (
    "inception_v3_augmented_data_ISIC2018_Training_Input_2023-04-21_50__36c.pt"
)
TRAINED_RESNET18_MODEL_2018: str = (
    "resnet18_augmented_data_ISIC2018_Training_Input_2023-04-21_25__995.pt"
)
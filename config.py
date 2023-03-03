# Importing the required libraries
from typing import Tuple
from torchvision import transforms # Import the transforms module from torchvision
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

# this should only be assigned when using the 2018 dataset
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
LEARNING_RATE: float = 0.0001
MOMENTUM: float = 0.9

# PARAMETERS - Scheduler
STEP_SIZE: int = 40
GAMMA: int = 0.1

# PARAMETERS - Dataloader
BATCH_SIZE: int = 64
VAL_BATCH_SIZE: int = 64
PIN_MEMORY_TRAIN_DATALOADER: bool = True
SHUFFLE_TRAIN_DATALOADER: bool = True
SHUFFLE_VAL_DATALOADER: bool = True

# PARAMETERS - Model Training
EPOCH_COUNT: int = 50

TRAIN_SPLIT_PERCENTAGE: float = 0.8
VAL_SPLIT_PERCENTAGE: float = 0.2
TEST_SPLIT_PERCENTAGE: float = 1 - TRAIN_SPLIT_PERCENTAGE - VAL_SPLIT_PERCENTAGE


# PARAMETERS - DATA AUGMENTATION
MIN_NUMBER_OF_EACH_CLASS: int = 3000
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


IS_2018_DATASET: bool = True
if "2019" in TRAIN_DATASET_LABELS and "2019" in TRAIN_DATASET_ROOT_DIR:
    IS_2018_DATASET = False
    assert (TRAIN_SPLIT_PERCENTAGE + TEST_SPLIT_PERCENTAGE + VAL_SPLIT_PERCENTAGE) == 1.0, "The total of train, validation and test should be equal to 1.0"
else:
    assert (TRAIN_SPLIT_PERCENTAGE + VAL_SPLIT_PERCENTAGE) == 1.0, "The total of train and validation should be equal to 1.0"



assert "2019" in TRAIN_DATASET_LABELS and "2019" not in TRAIN_DATASET_ROOT_DIR, "Need to have 2019 dataset to 2019 labels"
assert "2018" in TRAIN_DATASET_LABELS and "2018" not in TRAIN_DATASET_ROOT_DIR, "Need to have 2018 dataset to 2018 labels"
# Import the required libraries
import sys
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from config import RESNET18_MODEL_NAME, TEST_2018_LABELS, TEST_2018_ROOT_DIR, TRAIN_2018_LABELS, TRAIN_2018_ROOT_DIR
from config import BATCH_SIZE, EPOCH_COUNT, TRAIN_NROWS, TEST_NROWS, IMAGE_FILE_TYPE, IMAGENET_MEAN, IMAGENET_STD, RESNET18_PIXEL_SIZE, LEARNING_RATE, MOMENTUM
from customDataset import ISICDataset
from misc_helper import save_model_to_file
from run_helper import test_model, train_model_finetuning

# Set the randomness seeds
torch.manual_seed(42)
np.random.seed(42)

print("Setting up preprocess transforms...")
# Define image pre-processing steps
preprocess_resnet18 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(RESNET18_PIXEL_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Load the datasets
print("Loading in datasets...")
train_dataset_2018_resnet18 = ISICDataset(
    csv_file=TRAIN_2018_LABELS, 
    root_dir=TRAIN_2018_ROOT_DIR, 
    transform=preprocess_resnet18,
    image_file_type=IMAGE_FILE_TYPE,
    nrows=TRAIN_NROWS
)
test_dataset_2018_resnet18 = ISICDataset(
    csv_file=TEST_2018_LABELS, 
    root_dir=TEST_2018_ROOT_DIR, 
    transform=preprocess_resnet18,
    image_file_type=IMAGE_FILE_TYPE,
    nrows=TEST_NROWS
)

# Define the train data loader
print("Define the train data loader...")
data_loader_train_2018 = torch.utils.data.DataLoader(train_dataset_2018_resnet18, batch_size=BATCH_SIZE, shuffle=True)

# Load the pretrained Resnet-18 model
print("Load the pretrained Resnet-18 model...")
model_resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Define the criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_resnet18.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

print("Start training model...")
model_resnet18 = train_model_finetuning(
    model_resnet18, 
    train_dataset_2018_resnet18, 
    data_loader_train_2018,
    criterion,
    optimizer,
    model_name="resnet18",
    epoch_count=EPOCH_COUNT
)

# save the model to file
train_set_name: str = next(iter(TEST_2018_ROOT_DIR.split("/")), None)
save_model_to_file(model_resnet18, RESNET18_MODEL_NAME, train_set_name, models_dir="models")


# Define the test data loader
print("Define the test data loader...")
data_loader_test_2018 = torch.utils.data.DataLoader(test_dataset_2018_resnet18, batch_size=BATCH_SIZE, shuffle=False)



# Test the model's performance
print("Test the model's performance...")
test_model(model_resnet18, test_dataset_2018_resnet18, data_loader_test_2018, model_name="resnet18")

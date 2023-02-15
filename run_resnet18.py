#########################################################
# ============ IMPORT REQUIRED LIBRARIES ================
#########################################################
import torch # Import the Pytorch library
import torchvision # Import the torchvision library
from torchvision import datasets, transforms # Import the transforms module from torchvision


import numpy as np
from PIL import Image
from config import TEST_2018_LABELS, TEST_2018_ROOT_DIR, TRAIN_2018_LABELS, TRAIN_2018_ROOT_DIR # Import the Image module from the Python Imaging Library (PIL)

from customDataset import ISICDataset
from run_helper import train_model_finetuning # Custom dataset class


print("Setting up preprocess transforms...")
# Define the image pre-processing steps
preprocess_resnet18 = transforms.Compose([
    transforms.ToPILImage(), # Removes potential errors in Inception V3, may need it here also
    transforms.Resize(256),  # Resize the image to 256x256 pixels
    transforms.CenterCrop(224), # Crop the image to 224x224 pixels (removing any extra pixels)
    transforms.ToTensor(), # Convert the image to a Pytorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image using the pre-trained model's mean and standard deviation
])


# ================= DATASETS ================= #
print("Loading in datasets...")
# Training set 2018 - custom class
train_dataset_2018_resnet18 = ISICDataset(
    csv_file=TRAIN_2018_LABELS, 
    root_dir=TRAIN_2018_ROOT_DIR, 
    transform=preprocess_resnet18,
    image_file_type="jpg",
    #nrows=5000 # defines the number of rows used, utilized this for testing purposes
    )

# Test set 2018 - custom class
test_dataset_2018_resnet18 = ISICDataset(
    csv_file=TEST_2018_LABELS, 
    root_dir=TEST_2018_ROOT_DIR, 
    transform=preprocess_resnet18,
    image_file_type="jpg",
    # nrows=100 # defines the number of rows used, utilized this for testing purposes
    )


# Define the data loader
print("Define the data loader...")
data_loader_train_2018 = torch.utils.data.DataLoader(train_dataset_2018_resnet18, batch_size=32, shuffle=True)

# Load the pretrained Resnet-18 model
print("Load the pretrained Resnet-18 model...")
model_resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True);

# Define criterion and optimizer -> do not use adam, since learning rate is so small
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_resnet18.fc.parameters(), lr=0.001, momentum=0.9)


print("Start training model...")
model_resnet18 = train_model_finetuning(
    model_resnet18, 
    train_dataset_2018_resnet18, 
    data_loader_train_2018,
    criterion,
    optimizer,
    epoch_count=100
    )



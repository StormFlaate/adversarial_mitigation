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
# Define image pre-processing steps
preprocess_inceptionv3 = transforms.Compose([
    transforms.ToPILImage(), # Removes error
    transforms.Resize(299), # Resize the image to 299x299 pixels
    transforms.CenterCrop(299), # Crop the image to 299x299 pixels (removing any extra pixels)
    transforms.ToTensor(), # Convert the image to a Pytorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image using the pre-trained model's mean and standard deviation
])


# ================= DATASETS ================= #
print("Loading in datasets...")
# Training set 2018 - custom class
train_dataset_2018_inceptionv3 = ISICDataset(
    csv_file=TRAIN_2018_LABELS, 
    root_dir=TRAIN_2018_ROOT_DIR, 
    transform=preprocess_inceptionv3,
    image_file_type="jpg",
    #nrows=5000 # defines the number of rows used, utilized this for testing purposes
    )

# Test set 2018 - custom class
test_dataset_2018_inceptionv3 = ISICDataset(
    csv_file=TEST_2018_LABELS, 
    root_dir=TEST_2018_ROOT_DIR, 
    transform=preprocess_inceptionv3,
    image_file_type="jpg",
    # nrows=100 # defines the number of rows used, utilized this for testing purposes
    )


# Define the data loader
print("Define the data loader...")
data_loader_train_2018 = torch.utils.data.DataLoader(train_dataset_2018_inceptionv3, batch_size=32, shuffle=True)

# Load the pretrained Inception v3 model
print("Load the pretrained Inception v3 model...")
model_inceptionv3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

# Define criterion and optimizer -> do not use adam, since learning rate is so small
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_inceptionv3.fc.parameters(), lr=0.001, momentum=0.9)


print("Start training model...")
model_inceptionv3 = train_model_finetuning(
    model_inceptionv3, 
    train_dataset_2018_inceptionv3, 
    data_loader_train_2018,
    criterion,
    optimizer,
    model_name="inceptionv3",
    epoch_count=100
    )



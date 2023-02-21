# Import the required libraries
import sys
import torch
import torchvision
from torch.utils.data import random_split, Dataset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from config import (
    GAMMA, MODEL_NAME, NUM_WORKERS, PIN_MEMORY_TRAIN_DATALOADER, PREPROCESS_TRANSFORM, 
    SHUFFLE_TRAIN_DATALOADER, 
    STEP_SIZE, TEST_DATASET_LABELS, TEST_DATASET_ROOT_DIR, SHUFFLE_VAL_DATALOADER,
    TRAIN_DATASET_LABELS, TRAIN_DATASET_ROOT_DIR, IMAGE_FILE_TYPE,
    TRAIN_NROWS, BATCH_SIZE, TEST_NROWS, LEARNING_RATE, MOMENTUM,
    EPOCH_COUNT, TRAIN_SPLIT_PERCENTAGE, VAL_SPLIT_PERCENTAGE)
from customDataset import ISICDataset
from misc_helper import save_model_and_parameters_to_file
from run_helper import test_model, train_model

if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn')

    # Set the randomness seeds
    torch.manual_seed(42)
    np.random.seed(42)


    # Load the datasets
    print("Loading datasets...")
    train_dataset_full = ISICDataset(
        csv_file=TRAIN_DATASET_LABELS, 
        root_dir=TRAIN_DATASET_ROOT_DIR, 
        transform=PREPROCESS_TRANSFORM,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TRAIN_NROWS
    )

    # Splits the dataset into train and validation
    train_dataset, val_dataset = random_split(train_dataset_full, [TRAIN_SPLIT_PERCENTAGE, VAL_SPLIT_PERCENTAGE])
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")

    test_dataset_full = ISICDataset(
        csv_file=TEST_DATASET_LABELS, 
        root_dir=TEST_DATASET_ROOT_DIR, 
        transform=PREPROCESS_TRANSFORM,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TEST_NROWS
    )


    # Define the data loaders
    print("Defining data loaders...")
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE_TRAIN_DATALOADER,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY_TRAIN_DATALOADER
    )

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE_VAL_DATALOADER,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY_TRAIN_DATALOADER
    )


    # Load the pretrained model
    print(f"Loading pretrained {MODEL_NAME} model...")
    cnn_model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)


    # Define criterion and optimizer
    print("Defining criterion and optimizer...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)


    # Train the model
    print("Training model...")
    cnn_model = train_model(
        cnn_model, 
        train_dataset, 
        train_data_loader,
        val_data_loader,
        criterion,
        optimizer,
        scheduler,
        model_name=MODEL_NAME,
        epoch_count=EPOCH_COUNT
    )

    # save the model to file
    save_model_and_parameters_to_file(cnn_model, MODEL_NAME, TRAIN_DATASET_ROOT_DIR, models_dir="models")


    # Define the test data loader
    print("Define the test data loader...")
    data_loader_test = torch.utils.data.DataLoader(test_dataset_full, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


    # Test the model's performance
    print("Testing the model's performance...")
    test_model(cnn_model, test_dataset_full, data_loader_test, model_name=MODEL_NAME)

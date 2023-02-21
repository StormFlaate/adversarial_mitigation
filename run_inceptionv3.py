# Import the required libraries
import sys
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from config import GAMMA, INCEPTIONV3_MODEL_NAME, INCEPTIONV3_PIXEL_SIZE, NUM_WORKERS, RESNET18_MODEL_NAME, STEP_SIZE, TEST_DATASET_LABELS, TEST_DATASET_ROOT_DIR, TRAIN_DATASET_LABELS, TRAIN_DATASET_ROOT_DIR
from config import BATCH_SIZE, EPOCH_COUNT, TRAIN_NROWS, TEST_NROWS, IMAGE_FILE_TYPE, IMAGENET_MEAN, IMAGENET_STD, RESNET18_PIXEL_SIZE, LEARNING_RATE, MOMENTUM
from customDataset import ISICDataset
from misc_helper import save_model_to_file
from run_helper import test_model, train_model

if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn')

    # Set the randomness seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Define image pre-processing steps
    preprocess_inceptionv3 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(INCEPTIONV3_PIXEL_SIZE),
        transforms.CenterCrop(INCEPTIONV3_PIXEL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Load the datasets
    print("Loading datasets...")
    train_dataset_inceptionv3 = ISICDataset(
        csv_file=TRAIN_DATASET_LABELS, 
        root_dir=TRAIN_DATASET_ROOT_DIR, 
        transform=preprocess_inceptionv3,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TRAIN_NROWS
    )
    test_dataset_inceptionv3 = ISICDataset(
        csv_file=TEST_DATASET_LABELS, 
        root_dir=TEST_DATASET_ROOT_DIR, 
        transform=preprocess_inceptionv3,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TEST_NROWS
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the data loaders
    print("Defining data loaders...")
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset_inceptionv3, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )



    # Load the pretrained Inception v3 model
    print("Loading pretrained Inception v3 model...")
    model_inceptionv3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights=True)

    # Define criterion and optimizer
    print("Defining criterion and optimizer...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_inceptionv3.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    # Train the model
    print("Training model...")
    model_inceptionv3 = train_model(
        model_inceptionv3, 
        train_dataset_inceptionv3, 
        train_data_loader,
        criterion,
        optimizer,
        scheduler,
        model_name=INCEPTIONV3_MODEL_NAME,
        epoch_count=EPOCH_COUNT
    )

    # save the model to file
    train_set_name: str = next(iter(TEST_DATASET_ROOT_DIR.split("/")), None)
    save_model_to_file(model_inceptionv3, INCEPTIONV3_MODEL_NAME, train_set_name, models_dir="models")

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset_inceptionv3, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True)

    # Test the model's performance
    print("Testing the model's performance...")
    test_model(model_inceptionv3, test_dataset_inceptionv3, data_loader_test, model_name=INCEPTIONV3_MODEL_NAME)

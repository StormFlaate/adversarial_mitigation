# Import the required libraries
import torch
import numpy as np
import multiprocessing as mp
from misc_helper import save_model_and_parameters_to_file
from train_model_helper import get_data_loaders, test_model, train_model
from config import (GAMMA, MODEL_NAME, RANDOM_SEED, STEP_SIZE, TRAIN_DATASET_ROOT_DIR,
                    LEARNING_RATE, MOMENTUM, EPOCH_COUNT)

if __name__ == '__main__':
    mp.freeze_support()
    mp.set_start_method('spawn')

    # Set the randomness seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


    # collect the 3 data loaders based on CONFIG.PY file
    train_data_loader, val_data_loader, test_data_loader = get_data_loaders()

    # Load the pretrained model
    print(f"Loading pretrained {MODEL_NAME} model...")
    cnn_model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)


    # Define criterion and optimizer
    print("Defining criterion and optimizer...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        cnn_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
        )
    # optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
        )

    # Train the model
    print("Training model...")
    cnn_model = train_model(
        cnn_model,
        train_data_loader,
        val_data_loader,
        criterion,
        optimizer,
        scheduler,
        model_name=MODEL_NAME,
        epoch_count=EPOCH_COUNT
    )

    # save the model to file
    save_model_and_parameters_to_file(
        cnn_model, MODEL_NAME, TRAIN_DATASET_ROOT_DIR, EPOCH_COUNT, models_dir="models"
        )


    # Test the model's performance
    print("Testing the model's performance...")
    test_model(cnn_model, test_data_loader, model_name=MODEL_NAME)

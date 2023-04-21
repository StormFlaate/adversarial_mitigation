# Import the required libraries
import argparse
import torch
import numpy as np
import multiprocessing as mp
from torchvision.models import inception_v3, resnet18
from torchvision.models import ResNet18_Weights
from torchvision.models import Inception_V3_Weights
from helper_functions.misc_helper import save_model_and_parameters_to_file
from helper_functions.train_model_helper import (
    get_data_loaders_2018,
    get_data_loaders_2019,
    test_model,
    train_model,
)
from torch.utils.tensorboard import SummaryWriter
from config import (
    GAMMA,
    INCEPTIONV3_MODEL_NAME,
    PREPROCESS_INCEPTIONV3,
    PREPROCESS_RESNET18,
    RANDOM_SEED,
    RESNET18_MODEL_NAME,
    STEP_SIZE,
    LEARNING_RATE,
    MOMENTUM,
    EPOCH_COUNT
)


def set_random_seeds():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)


def load_pretrained_model_inceptionv3_and_transform():
    print(f"Loading pretrained {INCEPTIONV3_MODEL_NAME} model...")
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=False)
    return (model,PREPROCESS_INCEPTIONV3)


def load_pretrained_model_resnet18_and_transform():
    print(f"Loading pretrained {RESNET18_MODEL_NAME} model...")
    return (
        resnet18(weights=ResNet18_Weights.DEFAULT),
        PREPROCESS_RESNET18
    )


def define_criterion_and_optimizer(model, learning_rate, momentum, step_size, gamma):
    print("Defining criterion and optimizer...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    return criterion, optimizer, scheduler

def check_model_name(model_name):
    assert model_name in {INCEPTIONV3_MODEL_NAME, RESNET18_MODEL_NAME}


def print_augmentation_status(use_augmented_data):
    if use_augmented_data:
        print("Using the augmented dataset...")
    else:
        print("Using the NON augmented dataset...")


def load_model_and_transform(model_name):
    if model_name == INCEPTIONV3_MODEL_NAME:
        return load_pretrained_model_inceptionv3_and_transform()
    elif model_name == RESNET18_MODEL_NAME:
        return load_pretrained_model_resnet18_and_transform()
    else:
        raise Exception("Need to choose a model architecture...")


def get_data_loaders_by_year(year, transform, use_augmented_data):
    if year == "2018":
        return get_data_loaders_2018(
            transform=transform,
            is_augmented_dataset=use_augmented_data
        )
    elif year == "2019":
        return get_data_loaders_2019(
            transform=transform,
            is_augmented_dataset=use_augmented_data
        )
    else:
        raise Exception("Need to choose dataset year...")

def load_train_and_save_model(
        model_name,
        year,
        learning_rate,
        momentum,
        step_size,
        gamma,
        epoch_count,
        use_augmented_data,
        writer
    ):

    cnn_model, transform = load_model_and_transform(model_name)
    *data_loaders, train_dataset_root_dir = get_data_loaders_by_year(year, transform, use_augmented_data)
    train_data_loader, validation_data_loader, test_data_loader = data_loaders

    criterion, optimizer, scheduler = define_criterion_and_optimizer(
        cnn_model,
        learning_rate,
        momentum,
        step_size,
        gamma
    )

    # Train the model
    print("Training model...")
    cnn_model = train_model(
        cnn_model,
        train_data_loader,
        validation_data_loader,
        criterion,
        optimizer,
        scheduler,
        train_dataset_root_dir,
        writer,
        model_name=model_name,
        epoch_count=epoch_count
    )

    # save the model to file
    save_model_and_parameters_to_file(
        cnn_model, model_name, train_dataset_root_dir, epoch_count, models_dir="models"
    )

    return cnn_model, train_data_loader, validation_data_loader, test_data_loader



def main(year, model_name, use_augmented_data):
    mp.freeze_support()
    mp.set_start_method('spawn')
    set_random_seeds()
    check_model_name(model_name)
    print_augmentation_status(use_augmented_data)
    writer = SummaryWriter()

    # Train the model with the best learning rate
    cnn_model, *_, test_data_loader = load_train_and_save_model(
        model_name,
        year,
        LEARNING_RATE,
        MOMENTUM,
        STEP_SIZE,
        GAMMA,
        EPOCH_COUNT,
        use_augmented_data,
        writer
    )

    # Test the model's performance
    print("Testing the model's performance...")
    test_model(cnn_model, test_data_loader, model_name=model_name)

    writer.close()

       


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model-name, dataset-year, possibility to use non-augmented dataset and add multiple learning rates."
    )
    # Add argument for the dataset year
    parser.add_argument(
        "--year",
        required=True,
        choices=["2018", "2019"],
        help="Dataset for which to perform training on (2018 or 2019)."
    )

    # Add argument for the model type
    parser.add_argument(
        "--model",
        required=True,
        choices=[INCEPTIONV3_MODEL_NAME, RESNET18_MODEL_NAME],
        help=(
            f"Model for which to perform training ({INCEPTIONV3_MODEL_NAME}"
            f" or {RESNET18_MODEL_NAME})"
        )
    )

    # Add argument to disable data augmentation
    parser.add_argument(
        "--not-augment",
        action="store_true",
        help="Disable the use of the data augmented dataset"
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.year, args.model, not args.not_augment)
    

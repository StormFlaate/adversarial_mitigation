# Import the required libraries
import argparse
import torch
import numpy as np
import multiprocessing as mp
from helper_functions.misc_helper import save_model_and_parameters_to_file
from helper_functions.train_model_helper import (
    get_data_loaders_2018,
    get_data_loaders_2019,
    test_model,
    train_model
)
from config import (
    GAMMA,
    INCEPTIONV3_MODEL_NAME,
    LEARNING_RATES,
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
    model = torch.hub.load(
        'pytorch/vision:v0.10.0', INCEPTIONV3_MODEL_NAME, pretrained=True
    )
    model.aux_logits = False
    
    return (model,PREPROCESS_INCEPTIONV3)


def load_pretrained_model_resnet18_and_transform():
    print(f"Loading pretrained {RESNET18_MODEL_NAME} model...")
    return (
        torch.hub.load(
            'pytorch/vision:v0.10.0', RESNET18_MODEL_NAME, pretrained=True
        ),
        PREPROCESS_RESNET18
    )


def define_criterion_and_optimizer(model, learning_rate, momentum, step_size, gamma):
    print("Defining criterion and optimizer...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
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

def main(year, model_name, use_augmented_data, learning_rates):
    mp.freeze_support()
    mp.set_start_method('spawn')
    set_random_seeds()

    for learning_rate in learning_rates:
        print(f"Learning rate: {learning_rate}")

        check_model_name(model_name)
        print_augmentation_status(use_augmented_data)
        cnn_model, transform = load_model_and_transform(model_name)
        *data_loaders, train_dataset_root_dir = get_data_loaders_by_year(year, transform, use_augmented_data)
        train_data_loader, validation_data_loader, test_data_loader = data_loaders

        criterion, optimizer, scheduler = define_criterion_and_optimizer(
            cnn_model,
            learning_rate,
            MOMENTUM,
            STEP_SIZE,
            GAMMA
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
            model_name=model_name,
            epoch_count=EPOCH_COUNT
        )

        # save the model to file
        save_model_and_parameters_to_file(
            cnn_model, model_name, train_dataset_root_dir, EPOCH_COUNT, models_dir="models"
        )

        # Test the model's performance
        print("Testing the model's performance...")
        test_model(cnn_model, test_data_loader, model_name=model_name)
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model-name, dataset-year and removing data augmentation"
    )
    parser.add_argument(
        "--year",
        required=True,
        choices=["2018", "2019"],
        help="Dataset for which to perform training on (2018 or 2019)."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[INCEPTIONV3_MODEL_NAME, RESNET18_MODEL_NAME],
        help=(
            f"Model for which to perform training ({INCEPTIONV3_MODEL_NAME}"
            f" or {RESNET18_MODEL_NAME})"
        )
    )

    # the default value is to use the augmented dataset
    parser.add_argument(
        "--not-augment",
        action="store_true",
        help="Disable the use of the data augmented dataset"
    )

    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[LEARNING_RATE],
        help="A list of learning rates for the optimizer, default is the learning rate in config.py file",
    )

    args = parser.parse_args()
    main(args.year, args.model, not args.not_augment, args.learning_rates)

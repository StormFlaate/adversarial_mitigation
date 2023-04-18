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




def load_pretrained_model_inceptionv3():
    print(f"Loading pretrained {INCEPTIONV3_MODEL_NAME} model...")
    return torch.hub.load(
        'pytorch/vision:v0.10.0', INCEPTIONV3_MODEL_NAME, pretrained=True
    )

def load_pretrained_model_resnet18():
    print(f"Loading pretrained {RESNET18_MODEL_NAME} model...")
    return torch.hub.load(
        'pytorch/vision:v0.10.0', RESNET18_MODEL_NAME, pretrained=True
    )

def define_criterion_and_optimizer(model):
    print("Defining criterion and optimizer...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )
    return criterion, optimizer, scheduler


def main(year, model_name, use_augmented_data):
    mp.freeze_support()
    mp.set_start_method('spawn')
    set_random_seeds()

    assert model_name in {INCEPTIONV3_MODEL_NAME, RESNET18_MODEL_NAME}
    if use_augmented_data:
        print("Using the augmented dataset...")
    else:
        print("Using the NON augmented dataset...")

    if model_name == INCEPTIONV3_MODEL_NAME:
        cnn_model = load_pretrained_model_inceptionv3()
    elif model_name == RESNET18_MODEL_NAME:
        cnn_model = load_pretrained_model_resnet18()
    else:
        raise Exception("Need to choose a model architecture...")

    if year == "2018":
        *data_loaders, train_dataset_root_dir = get_data_loaders_2018(
            is_augmented_dataset=use_augmented_data
        )
        train_data_loader, validation_data_loader, test_data_loader = data_loaders
    elif year == "2019":
        *data_loaders, train_dataset_root_dir = get_data_loaders_2019(
            is_augmented_dataset=use_augmented_data
        )
        train_data_loader, validation_data_loader, test_data_loader = data_loaders
    else:
        raise Exception("Need to choose dataset year...")

    criterion, optimizer, scheduler = define_criterion_and_optimizer(cnn_model)

    # Train the model
    print("Training model...")
    cnn_model = train_model(
        cnn_model,
        train_data_loader,
        validation_data_loader,
        criterion,
        optimizer,
        scheduler,
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

    parser.add_argument(
        "--not-augment",
        action="store_true",
        help="Disable the use of the data augmented dataset"
    )

    args = parser.parse_args()
    main(args.year, args.model, not args.not_augment)

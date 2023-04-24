import argparse
import numpy as np
import torch
import torchattacks
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import multiprocessing as mp
from config import PREPROCESS_INCEPTIONV3, PREPROCESS_RESNET18, RANDOM_SEED, RESNET18_MODEL_NAME

from helper_functions.adversarial_attacks_helper import (
    extract_kernels_from_resnet_architecture,
    assess_attack_and_log_distances,
    plot_colored_grid
)
from helper_functions.misc_helper import get_trained_or_default_model
from helper_functions.train_model_helper import get_data_loaders_by_year


def _initialize_model(model_name: str, model_file_name: str) -> torch.nn.Module:
    """
    Initialize the model.

    Returns:
        torch.nn.Module: Trained or default model.
    """
    print("get trained or default model...")
    return get_trained_or_default_model(
        model_name,
        model_file_name=model_file_name
    )

def _initialize_data_loader_inception_v3(year):
    return get_data_loaders_by_year(year, PREPROCESS_INCEPTIONV3, False)


def _initialize_data_loader_resnet18(year):
    return get_data_loaders_by_year(year, PREPROCESS_RESNET18, False)

def _initialize_device() -> torch.device:
    """
    Initialize the device.

    Returns:
        torch.device: The device to be used (CPU or GPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _print_overall_accuracy(
        correct_labels: np.ndarray | list,
        predicted_labels: np.ndarray | list,
        predicted_adversarial_labels: np.ndarray | list
    ) -> None:
    """
    Calculates and prints the overall accuracy and overall adversarial accuracy.

    Args:
        correct_labels (np.ndarray): The correct labels.
        predicted_labels (np.ndarray): The predicted labels.
        predicted_adversarial_labels (np.ndarray): The predicted adversarial labels.

    Returns:
        None
    """
    overall_accuracy = accuracy_score(correct_labels, predicted_labels)
    overall_adversarial_accuracy = accuracy_score(
        correct_labels, predicted_adversarial_labels
    )

    print("Overall accuracy: ", overall_accuracy)
    print("Overall adversarial accuracy: ", overall_adversarial_accuracy)



def main(year, model_file_name):
    mp.freeze_support()
    mp.set_start_method('spawn')
    # Set the randomness seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize empty lists
    log_distances: list = []
    correct_labels: list = []
    predicted_labels: list = []
    predicted_adversarial_labels: list = []
    torch.cuda.empty_cache()
    # Initialize setup
    train_data_loader, *_ = _initialize_data_loader_resnet18(year)
    model = _initialize_model(
        RESNET18_MODEL_NAME,
        model_file_name=model_file_name
    )
    device = _initialize_device()
    attack = torchattacks.FGSM(model, eps=2/255)

    # Initialize variables
    model_children: list = list(model.children()) # get all the model children as list
    _, conv_layers = extract_kernels_from_resnet_architecture(
            model_children)
    
    print("Length of convolutional layers: ", len(conv_layers))
    print(conv_layers)

    for i, (input, true_label) in tqdm(enumerate(train_data_loader)):

        label_results = assess_attack_and_log_distances(
            model, device, input, true_label, attack, conv_layers
        )
        log_distance, correct_label, predicted_label, adv_label = label_results

        [print(tensor.size()) for tensor in log_distance]

        # gets the tensors over to the cpu and then over to numpy
        log_distance: list = [tensor.detach().cpu().numpy() for tensor in log_distance]
        
        log_distances.append(log_distance)
        correct_labels.append(correct_label)
        predicted_labels.append(predicted_label)
        predicted_adversarial_labels.append(adv_label)
        
        if i >= 1:
            break
    
    
    print(log_distances)

    
    _print_overall_accuracy(
        correct_labels, predicted_labels, predicted_adversarial_labels
    )
    
    print(log_distances[0])
    plot_colored_grid(log_distances[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Model-name, dataset-year, possibility to use non-augmented dataset" 
            " and add multiple learning rates."
        )
    )
    # Add argument for the dataset year
    parser.add_argument(
        "--year",
        required=True,
        choices=["2018", "2019"],
        help="Dataset for which to perform training on (2018 or 2019)."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.year,
        "resnet18_augmented_data_ISIC2018_Training_Input_2023-04-21_50__a7f.pt"
    )

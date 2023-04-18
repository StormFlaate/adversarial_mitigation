from matplotlib import pyplot as plt
import numpy as np
import torch
import torchattacks
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from config import RANDOM_SEED

from helper_functions.adversarial_attacks_helper import (
    extract_kernels_from_resnet_architecture,
    assess_attack_and_log_distances
)
from helper_functions.misc_helper import get_trained_or_default_model
from helper_functions.train_model_helper import get_data_loaders


def _initialize_data_loaders() -> tuple:
    """
    Initialize data loaders.

    Returns:
        tuple: Train, validation and test data loaders.
    """
    print("get data loaders...")
    return get_data_loaders(batch_size=1, num_workers=1)


def _initialize_model() -> torch.nn.Module:
    """
    Initialize the model.

    Returns:
        torch.nn.Module: Trained or default model.
    """
    print("get trained or default model...")
    return get_trained_or_default_model(
        model_file_name="resnet18_augmented_data_ISIC2018_Training_Input_2023-03-08_50__bb6.pt"
    )


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



def main():
    # Set the randomness seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize empty lists
    log_distances: list = []
    correct_labels: list = []
    predicted_labels: list = []
    predicted_adversarial_labels: list = []
    
    # Initialize setup
    train_data_loader, _, _ = _initialize_data_loaders()
    model = _initialize_model()
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

        # gets the tensors over to the cpu and then over to numpy
        log_distance_array: np.array = np.array(
            [tensor.cpu().detach().numpy() for tensor in log_distance]
        )


        log_distances.append(log_distance_array)
        correct_labels.append(correct_label)
        predicted_labels.append(predicted_label)
        predicted_adversarial_labels.append(adv_label)
        
        if i >= 0:
            break

    _print_overall_accuracy(
        correct_labels, predicted_labels, predicted_adversarial_labels
    )

    log_distances_np = np.array(log_distances)


    print(log_distances_np[0][0][:10])
    print(log_distances_np[0][1][:10])
    print(log_distances_np)
    print(log_distances_np.shape)


if __name__ == '__main__':
    main()
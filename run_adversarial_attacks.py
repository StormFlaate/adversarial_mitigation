from matplotlib import pyplot as plt
import numpy as np
import torch
import torchattacks
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from config import RANDOM_SEED

from helper_functions.adversarial_attacks_helper import (
    extract_kernels_from_resnet_architecture,
    assess_attack_and_log_distances,
    plot_colored_grid
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

        [print(tensor.size()) for tensor in log_distance]

        # gets the tensors over to the cpu and then over to numpy
        log_distance: list = [tensor.detach().numpy() for tensor in log_distance]
        log_distance_array = np.array(log_distance)
        
        print(log_distance_array.shape)


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


    
    print(log_distances_np[0])
    print(log_distances_np.shape)
    plot_colored_grid(log_distances_np[0])


if __name__ == '__main__':
    # Example usage:
    # data = np.array([
    #     [-4.4240346, -3.85995, -3.863896,-4.4341908,-4.6712766,-4.1185102, -3.4920495, -3.7436595, -3.451645, -3.0995665],
    #     [-2.4328566, -2.2482922, -2.2400074, -2.3119617, -2.7472174, -4.0555816, -5.8380814, -6.6805916, -3.9911015, -2.6688745]
    # ])

    # plot_colored_grid(data)
    main()
import numpy as np
import torch
import torchattacks
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from helper_functions.adversarial_attacks_helper import (
    extract_feature_map_of_convolutional_layers,
    extract_kernels_from_resnet_architecture,
    generate_adversarial_input
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


def _process_batch(
        model: torch.nn.Module,
        device: torch.device,
        input: torch.Tensor,
        true_label: torch.Tensor,
        attack: torchattacks.attack, 
        conv_layers: list[torch.nn.Conv2d]
    ) -> tuple:
    """
    Process a single batch of data.

    Args:
        model (torch.nn.Module): The target model.
        device (torch.device): The device to be used.
        input (torch.Tensor): The input tensor.
        true_label (torch.Tensor): The true label tensor.
        attack (torchattacks.Attack): The attack method object.
        conv_layers: The convolutional layers for the current model.

    Returns:
        tuple: The true label, predicted label, and predicted adversarial label.
    """
    input = input.to(device)
    true_label = true_label.to(device)

    feature_map_before_attack = extract_feature_map_of_convolutional_layers(
        input, conv_layers)
    
    # perform adversarial attack on the input
    adversarial_input = generate_adversarial_input(input, true_label, attack)
    # evaluates the input using the trained model
    predicted_label = model(input)
    predicted_adversarial_label = model(adversarial_input)

    feature_map_after_attack = extract_feature_map_of_convolutional_layers(
        adversarial_input, conv_layers)

    logarithmic_distances = _calculate_logarithmic_distances(
        feature_map_before_attack, feature_map_after_attack)
    print(
        "Logarithmic distances between feature map before and after adversarial attack:"
        , logarithmic_distances
    )

    return (
        np.argmax(true_label.detach().cpu().numpy()),
        np.argmax(predicted_label.detach().cpu().numpy()),
        np.argmax(predicted_adversarial_label.detach().cpu().numpy())
    )

def _calculate_logarithmic_distances(before_attack, after_attack):
    """
    Calculate logarithmic distances between the feature maps before and after the attack.

    Args:
        before_attack (torch.Tensor): The feature map before the attack.
        after_attack (torch.Tensor): The feature map after the attack.
    
    Returns:
        A list of logarithmic distances for each layer.
    """
    distances = []

    for weights_before_attack, weights_after_attack in zip(before_attack, after_attack):
        flat_weights_before_attack = weights_before_attack.view(-1)
        flat_weights_after_attack = weights_after_attack.view(-1)
        difference = flat_weights_after_attack - flat_weights_before_attack
        logarithmic_distance = torch.mean(torch.log(torch.abs(difference)))
        distances.append(logarithmic_distance.item())

    return distances


def main():
    train_data_loader, _, _ = _initialize_data_loaders()
    model = _initialize_model()
    attack = torchattacks.FGSM(model, eps=2/255)
    type(attack)
    device = _initialize_device()

    model_children: list = list(model.children()) # get all the model children as list
    model_weights, conv_layers = extract_kernels_from_resnet_architecture(
            model_children)
    correct_labels = []
    predicted_labels = []
    predicted_adversarial_labels = []

    for _, (input, true_label) in tqdm(enumerate(train_data_loader)):
        
        correct_label, predicted_label, predicted_adversarial_label = _process_batch(
            model, device, input, true_label, attack, conv_layers
        )

        correct_labels.append(correct_label)
        predicted_labels.append(predicted_label)
        predicted_adversarial_labels.append(predicted_adversarial_label)
        break
        

    overall_accuracy = accuracy_score(correct_labels, predicted_labels)
    overall_adversarial_accuracy = accuracy_score(
        correct_labels, predicted_adversarial_labels
    )

    print("Overall accuracy: ", overall_accuracy)
    print("Overall adversarial accuracy: ", overall_adversarial_accuracy)


if __name__ == '__main__':
    main()
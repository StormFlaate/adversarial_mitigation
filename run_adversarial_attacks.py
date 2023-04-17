import numpy as np
import torch
import torchattacks
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from helper_functions.adversarial_attacks_helper import (
    extract_feature_map_of_convolutional_layers,
    generate_adversarial_input,
    visualize_feature_map_of_convolutional_layers
)
from helper_functions.misc_helper import get_trained_or_default_model
from helper_functions.train_model_helper import get_data_loaders

def _initialize_data_loaders():
    print("get data loaders...")
    return get_data_loaders(batch_size=1, num_workers=1)

def _initialize_model():
    print("get trained or default model...")
    return get_trained_or_default_model(
        model_file_name="resnet18_augmented_data_ISIC2018_Training_Input_2023-03-08_50__bb6.pt"
    )

def _initialize_attack(model):
    return torchattacks.FGSM(model, eps=2/255)

def _initialize_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _process_batch(model, device, input, true_label, attack):
    input = input.to(device)
    true_label = true_label.to(device)

    before_attack = extract_feature_map_of_convolutional_layers(input, model)
    visualize_feature_map_of_convolutional_layers(before_attack, "before")

    adversarial_input = generate_adversarial_input(input, true_label, attack)
    predicted_label = model(input)
    predicted_adversarial_label = model(adversarial_input)

    after_attack = extract_feature_map_of_convolutional_layers(adversarial_input, model)
    visualize_feature_map_of_convolutional_layers(after_attack, "after")

    logarithmic_distances = _calculate_logarithmic_distances(before_attack, after_attack)
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
    distances = []

    for weights_before_attack, weights_after_attack in zip(before_attack, after_attack):
        flat_weights_before_attack = weights_before_attack.view(-1)
        flat_weights_after_attack = weights_after_attack.view(-1)
        difference = flat_weights_after_attack - flat_weights_before_attack
        logarithmic_distance = torch.mean(torch.log(torch.abs(difference) + 1e-8))
        distances.append(logarithmic_distance.item())

    return distances

def main():
    train_data_loader, _, _ = _initialize_data_loaders()
    model = _initialize_model()
    attack = _initialize_attack(model)
    device = _initialize_device()

    correct_labels = []
    predicted_labels = []
    predicted_adversarial_labels = []

    for _, (input, true_label) in tqdm(enumerate(train_data_loader)):
        correct_label, predicted_label, predicted_adversarial_label = _process_batch(
            model, device, input, true_label, attack
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
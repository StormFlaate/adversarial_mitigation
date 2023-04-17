from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torchattacks
from tqdm import tqdm
from helper_functions.adversarial_attacks_helper import (
    extract_kernels_from_resnet_architecture,
    generate_adversarial_input
)
from helper_functions.misc_helper import get_trained_or_default_model
from helper_functions.train_model_helper import get_data_loaders

print("get data loaders...")
train_data_loader, val_data_loader, test_data_loader = get_data_loaders(
    batch_size=1, 
    num_workers=1)

print("get trained or deafult model...")
model = get_trained_or_default_model(
    model_file_name="resnet18_augmented_data_ISIC2018_Training_Input_2023-03-08_50__bb6.pt")


attack = torchattacks.FGSM(model, eps=2/255)

correct_labels: list[int] = []
predicted_labels: list[int] = []
predicted_adversarial_labels: list[int] = []
model_weights: list = [] # we will save the conv layer weights in this list
conv_layers: list = [] # we will save the 49 conv layers in this list
model_children: list = list(model.children()) # get all the model children as list


for index, (input, true_label) in  tqdm(enumerate(train_data_loader)):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    true_label = true_label.to(device)  

    model_weights, conv_layers = extract_kernels_from_resnet_architecture(
        list(model.children()), model_weights, conv_layers
    )
    weights_before_attack = model_weights[0]

    adversarial_input = generate_adversarial_input(input, true_label, attack)

    # predicting with adversarial and benign input    
    predicted_label = model(input)
    predicted_adversarial_label = model(adversarial_input)

    model_weights, conv_layers = extract_kernels_from_resnet_architecture(
        list(model.children()), model_weights, conv_layers
    )
    weights_after_attack = model_weights[0]

    # Flatten tensors
    flat_weights_before_attack = weights_before_attack.view(-1)
    flat_weights_after_attack = weights_after_attack.view(-1)

    # Calculate element-wise difference
    difference = flat_weights_after_attack - flat_weights_before_attack

    # Calculate logarithmic distance
    logarithmic_distance = torch.mean(torch.log(torch.abs(difference) + 1e-8))

    print(
        "Logarithmic distance between model weights before and after adversarial attack:",
        logarithmic_distance.item()
    )


    # take a look at the conv layers and the respective weights
    # for weight, conv in zip(model_weights, conv_layers):
    #     # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    #     print(f"CONV: {conv} ====> SHAPE: {weight.shape}")


    np_true_label = true_label.detach().cpu().numpy()
    np_predicted_label = predicted_label.detach().cpu().numpy()
    np_predicted_adversarial_label = predicted_adversarial_label.detach().cpu().numpy()
    
    correct_argmax_label = np.argmax(np_true_label)
    predicted_argmax_label = np.argmax(np_predicted_label)
    predicted_adversarial_argmax_label = np.argmax(np_predicted_adversarial_label)

    correct_labels.append(correct_argmax_label)
    predicted_labels.append(predicted_argmax_label)
    predicted_adversarial_labels.append(predicted_adversarial_argmax_label)

    
    break




overall_accuracy = accuracy_score(correct_labels, predicted_labels)

overall_adversarial_accuracy = accuracy_score(
    correct_labels,
    predicted_adversarial_labels
)

print("Overall accuracy: ", overall_accuracy)
print("Overall adversarial accuracy: ", overall_adversarial_accuracy)
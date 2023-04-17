from typing import Iterator
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


def generate_adversarial_input(
        input,
        label, 
        attack
        ) -> tuple[torch.tensor, torch.tensor]:
    """
        Applies adversarial attack to the input, creating an adversarial example.

        Args:
            input: input image or tensor
            label: correct label, 1-dimensional format [0,1,0,...]
            attack: specific attack form torchattacks

        Returns:
            (adversarial_input, label): attack applied to input and the correct label
    """
    # Move inputs and labels to the specified device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    label = label.to(device)    

    # turns 1-dimensional list into 0-dimensional scalar, needed for attack
    label_argmax = torch.argmax(label, 1)

    # generate the adver
    adversarial_input = attack(input, label_argmax)
    
    return adversarial_input


def extract_kernels_from_resnet_architecture(
    model_children: list[nn.Module],
    model_weights: list[torch.Tensor],
    conv_layers: list[nn.Conv2d]
    ) -> tuple[list[torch.Tensor], list[nn.Conv2d]]:
    """
    Extracts the kernel weights and convolutional layers from a ResNet architecture.

    Args:
        model_children (List[nn.Module]): A list of child modules from the ResNet model.
        model_weights (List[torch.Tensor]): A list to store the weights of the
            convolutional layers.
        conv_layers (List[nn.Conv2d]): A list to store the convolutional layers.

    Returns:
        Tuple[List[torch.Tensor], List[nn.Conv2d]]: A tuple containing two lists:
            1. The weights of the extracted convolutional layers.
            2. The extracted convolutional layers themselves.
    """
    # Initialize a counter to keep track of the number of convolutional layers
    counter = 0 

    # Iterate through the model's child modules
    for i in range(len(model_children)):
        # Check if the current child module is a convolutional layer
        if type(model_children[i]) == nn.Conv2d:
            # Increment the counter for each convolutional layer found
            counter += 1
            # Append the current layer's weights to the model_weights list
            model_weights.append(model_children[i].weight)
            # Append the current convolutional layer to the conv_layers list
            conv_layers.append(model_children[i])

        # Check if the current child module is a sequential layer
        elif type(model_children[i]) == nn.Sequential:
            # Iterate through the sub-modules within the sequential layer
            for j in range(len(model_children[i])):
                # Iterate through the children of each sub-module
                for child in model_children[i][j].children():
                    # Check if the current child is a convolutional layer
                    if type(child) == nn.Conv2d:
                        counter += 1
                        # Append the current layer's weights to the model_weights list
                        model_weights.append(child.weight)
                        # Append the current convolutional layer to the conv_layers list
                        conv_layers.append(child)

    # Print the total number of convolutional layers found
    print(f"Total convolutional layers: {counter}")

    # Return the updated model_weights and conv_layers lists as a tuple
    return model_weights, conv_layers


def extract_feature_map_of_convolutional_layers(
        input_tensor: torch.Tensor,
        conv_layers: list[nn.Conv2d]
    ) -> list[torch.Tensor]:
        # pass the image through all the layers
    results = [conv_layers[0](input_tensor)]

    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    
    return results

def visualize_feature_map_of_convolutional_layers(
        convolutional_outputs: list[torch.Tensor],
        file_name: str
    ) -> None:
    # visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
    for num_layer in range(len(convolutional_outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = convolutional_outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64: # we will visualize only 8x8 blocks from each layer
                break
            filter = filter.cpu()
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter)
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"./{file_name}_layer_{num_layer}.png")
        # plt.show()
        plt.close()
import math
from typing import Iterator
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torchattacks


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
    ) -> tuple[list[torch.Tensor], list[nn.Conv2d]]:
    """
    Extracts the kernel weights and convolutional layers from a ResNet architecture.

    Args:
        model_children (List[nn.Module]): A list of child modules from the ResNet model.

    Returns:
        Tuple[List[torch.Tensor], List[nn.Conv2d]]: A tuple containing two lists:
            1. The weights of the extracted convolutional layers.
            2. The extracted convolutional layers themselves.
    """

    model_weights = []
    conv_layers = []
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
    """
    Extracts the feature maps of a list of convolutional layers applied to an input
    tensor.

    Args:
        input_tensor: The input tensor to pass through the convolutional layers.
        conv_layers: A list of convolutional layers to apply to the input tensor.

    Returns:
        A list containing the feature maps of each convolutional layer applied to the
            input tensor.

    Raises:
        TypeError: If the input_tensor is not a torch.Tensor, or if conv_layers is not a
            list of nn.Conv2d layers.

    """
        # pass the image through all the layers
    results = [conv_layers[0](input_tensor)]

    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    
    return results

def visualize_feature_map_of_convolutional_layers(
        convolutional_outputs: list[torch.Tensor],
        file_name_prefix: str
    ) -> None:
    """Visualizes the feature maps of a list of convolutional layers.

    Args:
        convolutional_outputs: A list containing the feature maps of each convolutional
            layer.
        file_name_prefix: The base name to use when saving the visualizations of the
            feature maps.

    Returns:
        None

    Raises:
        TypeError: If the convolutional_outputs is not a list of torch.Tensor objects, 
            or if file_name_prefix is not a string.
    """
    # visualize features from each layer
    for num_layer in range(len(convolutional_outputs)):
        layer_viz = convolutional_outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        num_filters = layer_viz.size(0)

        # Calculate the dimensions of the grid based on the number of filters
        grid_size = int(math.ceil(math.sqrt(num_filters)))

        plt.figure(figsize=(grid_size * 3, grid_size * 3))

        for i, filter in enumerate(layer_viz):
            filter = filter.cpu()
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(filter)
            plt.axis("off")
        
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"./{file_name_prefix}_layer_{num_layer}.png")
        # plt.show()
        plt.close()


def assess_attack_and_log_distances(
        model: torch.nn.Module,
        device: torch.device,
        input: torch.Tensor,
        true_label: torch.Tensor,
        attack: torchattacks.attack, 
        conv_layers: list[torch.nn.Conv2d]
    ) -> tuple[list[torch.Tensor], np.intp, np.intp, np.intp]:
    """
    Assesses the attack before and after the pertubation of the input image, calculating
    log distance and what the model evaluates clean and pertubated input as.

    Args:
        model (torch.nn.Module): The target model.
        device (torch.device): The device to be used.
        input (torch.Tensor): The input tensor.
        true_label (torch.Tensor): The true label tensor.
        attack (torchattacks.Attack): The attack method object.
        conv_layers: The convolutional layers for the current model.

    Returns:
        The logarithmic distances between feature maps, true label, predicted label,
            and predicted adversarial label.
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

    logarithmic_distances = calculate_logarithmic_distances(
        feature_map_before_attack, feature_map_after_attack)

    return (
        logarithmic_distances,
        np.argmax(true_label.detach().cpu().numpy()),
        np.argmax(predicted_label.detach().cpu().numpy()),
        np.argmax(predicted_adversarial_label.detach().cpu().numpy())
    )


def calculate_logarithmic_distances(
        before_attack: list[torch.Tensor],
        after_attack: list[torch.Tensor]
    ) -> list[torch.Tensor]:
    """
    Calculate logarithmic distances between the feature maps before and after the attack.

    Args:
        before_attack (torch.Tensor): The feature map before the attack.
        after_attack (torch.Tensor): The feature map after the attack.
    
    Returns:
        A list of logarithmic distances for each layer.
    """
    distances = []

    for filters_before_attack, filters_after_attack in zip(before_attack, after_attack):
        assert filters_before_attack.shape == filters_after_attack.shape
        
        original_shape: torch.Size = filters_before_attack.shape

        difference = filters_after_attack.view(-1) - filters_before_attack.view(-1)
        logarithmic_distance_1_dim = torch.log(torch.abs(difference))

        # will ensure that the values that are 0 are changed to 0 instead of inf/-inf
        finite_mask = torch.isfinite(logarithmic_distance_1_dim)
        logarithmic_distance_1_dim[~finite_mask] = 0  # Set non-real values to 0

        logarithmic_distance_reshaped = logarithmic_distance_1_dim.reshape(
            original_shape
        )

        mean_logarithmic_distance = torch.mean(
            logarithmic_distance_reshaped, dim=(2, 3))

        print(mean_logarithmic_distance.shape)
        distances.append(mean_logarithmic_distance)

        # # Find the indices of the k feature maps with the greatest mean logarithmic distance
        # most_changed_indices = sorted(range(len(mean_logarithmic_distances)),
        #                           key=lambda i: mean_logarithmic_distances[i],
        #                           reverse=True)[:k]

    return distances


def plot_colored_grid(data: list[np.array], color_map='viridis'):
    nrows = len(data)
    max_ncols = max(arr.ndim for arr in data)
    print("max_ncols", max_ncols)
    fig, ax = plt.subplots(figsize=(max_ncols, nrows))

    # Normalize the data to map colors in the color map
    min_value = min(_flatten_list(data))
    max_value = max(_flatten_list(data))
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

    # Get the colormap object from the colormap name
    cmap = cm.get_cmap(color_map)

    for i in range(nrows):
        ncols = data[i].ndim
        for j in range(ncols):
            rect = plt.Rectangle(
                (j, i), 1, 1, facecolor=cmap(norm(data[i][j])), edgecolor='k'
            )
            ax.add_patch(rect)

    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - 0.5, minor=True)

    # Remove the major gridlines
    ax.grid(which='major', visible=False)

    # Set axis limits
    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)

    # Remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig("./test_images/colored_grid.png")
    plt.close()



def _flatten_list(list_of_arrays: list[np.array]
    ) -> np.array:
    # Stack the arrays in the list horizontally
    stacked_array = np.hstack(list_of_arrays)
    
    # Flatten the stacked array into a 1D array
    flattened_array = stacked_array.flatten()
    
    return flattened_array

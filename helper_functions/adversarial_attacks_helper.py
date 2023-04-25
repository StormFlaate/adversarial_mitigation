import math
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torchattacks

from config import INCEPTIONV3_MODEL_NAME, RESNET18_MODEL_NAME



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


def assess_attack_and_log_distances(
        model: torch.nn.Module,
        device: torch.device,
        input: torch.Tensor,
        true_label: torch.Tensor,
        attack: torchattacks.attack, 
        model_name: str
    ) -> tuple[list[torch.Tensor], np.intp, np.intp, np.intp]:
    """
    This is a function in Python that assesses the effect of an attack on an input
    image. It calculates the logarithmic distance between feature maps, true label,
    predicted label, and predicted adversarial label.

    Args:
        model (torch.nn.Module): The target model.
        device (torch.device): The device to be used.
        input (torch.Tensor): The input tensor.
        true_label (torch.Tensor): The true label tensor.
        attack (torchattacks.Attack): The attack method object.
        model_name (str): The name of the model used in the current attack.

    Returns:

    tuple: A tuple of logarithmic distances between feature maps, true label, predicted
        label, and predicted adversarial label.
    """
    input = input.to(device)
    true_label = true_label.to(device)

    feature_map_before_attack = get_feature_maps(input, model, model_name)
        
    # perform adversarial attack on the input
    adversarial_input = generate_adversarial_input(input, true_label, attack)

    # evaluates the input using the trained model
    predicted_label = model(input)
    predicted_adversarial_label = model(adversarial_input)

    feature_map_after_attack = get_feature_maps(input, model, model_name)

    logarithmic_distances = calculate_log_distances(
        feature_map_before_attack, feature_map_after_attack)

    return (
        logarithmic_distances,
        np.argmax(true_label.detach().cpu().numpy()),
        np.argmax(predicted_label.detach().cpu().numpy()),
        np.argmax(predicted_adversarial_label.detach().cpu().numpy())
    )



def calculate_log_distances(a_list: list[torch.Tensor], b_list: list[torch.Tensor]):
    """Calculate the log distance between two lists of feature maps.

    Args:
        a_list (list[torch.Tensor]): List of feature maps of shape (N, C, H, W).
        b_list (list[torch.Tensor]): List of feature maps of shape (N, C, H, W).

    Returns:
        float: List of log distance between the two feature maps.
    """
    log_distances:list = []
    for a, b in zip(a_list, b_list):
        # Flatten the feature maps and compute their L2 distance
        a = torch.flatten(a, start_dim=1)
        b = torch.flatten(b, start_dim=1)
        l2_distance = torch.norm(a - b, p=2, dim=1)

        # Compute the log distance
        log_distance = torch.log(l2_distance.mean())

        # will ensure that the values that are 0 are changed to 0 instead of inf/-inf
        finite_mask = torch.isfinite(log_distance)
        log_distance[~finite_mask] = 0  # Set non-real values to 0

        log_distances.append(log_distance.item())

    return log_distances




def _flatten_list(list_of_arrays: list[np.array]) -> np.array:
    # Stack the arrays in the list horizontally
    stacked_array = np.hstack(list_of_arrays)
    
    # Flatten the stacked array into a 1D array
    flattened_array = stacked_array.flatten()
    
    return flattened_array

def _get_normalize_function(row):
    # Normalize the data to map colors in the color map
    min_value = min(row)
    max_value = max(row)
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

    return norm


def get_feature_maps(input, model, model_name):
    """Get feature maps from a given model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torch.nn.Module): Model to extract feature maps from.
        model_name (str): Name of the model to extract feature maps from.

    Returns:
        List[torch.Tensor]: List of feature maps of shape (N, C, H, W).
    """
    if model_name == INCEPTIONV3_MODEL_NAME:
        return _get_feature_maps_inception_v3(input, model)
    elif model_name == RESNET18_MODEL_NAME:
        return _get_feature_maps_resnet18(input, model)
    else:
        raise Exception("Not valid model name")


def _get_feature_maps_inception_v3(input, model):
    """Get feature maps from InceptionV3 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torch.nn.Module): InceptionV3 model.

    Returns:
        List[torch.Tensor]: List of feature maps of shape (N, C, H, W).
    """
    feature_maps = []

    def hook(module, input, output):
        feature_maps.append(output.detach())

    # List of InceptionV3 layers to extract feature maps from
    layers = [
        model.Conv2d_1a_3x3,
        model.Conv2d_2a_3x3,
        model.Conv2d_2b_3x3,
        model.Conv2d_3b_1x1,
        model.Conv2d_4a_3x3,
        model.Mixed_5b,
        model.Mixed_5c,
        model.Mixed_5d,
        model.Mixed_6a,
        model.Mixed_6b,
        model.Mixed_6c,
        model.Mixed_6d,
        model.Mixed_6e,
        model.Mixed_7a,
        model.Mixed_7b,
        model.Mixed_7c
    ]
    # Register hook on each layer
    handles = [layer.register_forward_hook(hook) for layer in layers]

    _ = model(input)

    # Remove hook from each layer
    for handle in handles:
        handle.remove()

    return feature_maps


def _get_feature_maps_resnet18(input, model):
    """Get feature maps from ResNet18 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torch.nn.Module): ResNet18 model.

    Returns:
        List[torch.Tensor]: List of feature maps of shape (N, C, H, W).
    """
    feature_maps = []

    def hook(module, input, output):
        feature_maps.append(output.detach())

    # List of ResNet18 layers to extract feature maps from
    layers = [
        module for _, module in model.named_children() 
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Sequential))
    ]
    # Register hook on each layer
    handles = [layer.register_forward_hook(hook) for layer in layers]

    _ = model(input)

    # Remove hook from each layer
    for handle in handles:
        handle.remove()

    return feature_maps


def visualize_feature_maps(feature_maps, ncols=8, output_dir: str="./test_images/"):
    """Visualize feature maps.

    Args:
        feature_maps (List[torch.Tensor]): List of feature maps of shape (N, C, H, W).
        ncols (int, optional): Number of columns to display. Default is 8.
    """
    nrows = len(feature_maps)
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))

    for row in range(nrows):
        for col in range(ncols):
            axs[row, col].imshow(feature_maps[row][0, col].cpu().numpy())
            axs[row, col].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Save the images with different names based on the number of columns
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"colored_grid_{ncols}_columns.png"))
    plt.close()


def old_calculate_logarithmic_distances(
        before_attack: list[torch.Tensor],
        after_attack: list[torch.Tensor]
    ) -> list[torch.Tensor]:
    """
    Calculate logarithmic distances between the feature maps before and after the
    attack.

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


    return distances



def old_plot_colored_grid(data: list[np.array], color_map='viridis'):
    nrows = len(data)

    # Group rows by the number of columns
    grouped_data = {}
    for filter_index, arr in enumerate(data):
        ncols = arr.shape[1]
        if ncols in grouped_data:
            grouped_data[ncols].append((filter_index, arr))
        else:
            grouped_data[ncols] = [(filter_index, arr)]

    for ncols, grouped_rows in grouped_data.items():
        nrows = len(grouped_rows)
        fig, ax = plt.subplots(figsize=(ncols, nrows))

        # Get the colormap object from the colormap name
        cmap = cm.get_cmap(color_map)

        for i, (filter_index, arr) in enumerate(grouped_rows):
            current_row = arr.flatten()
            norm = _get_normalize_function(current_row)

            for j in range(ncols):
                rect = plt.Rectangle(
                    (j, i), 1, 1, facecolor=cmap(norm(current_row[j])), edgecolor='k'
                )
                ax.add_patch(rect)

                # Add column index as a tick label at the bottom of the grid
                if i == nrows - 1:  # Only add labels for the last row
                    ax.text(j+0.5, -0.5, str(j), ha='center', va='top')

            # Add filter index as a tick label to the left of the row
            ax.text(-0.5, i+0.5, str(filter_index), ha='right', va='center')

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

        # Save the images with different names based on the number of columns
        output_dir = "./test_images/"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"colored_grid_{ncols}_columns.png"))
        plt.close()



def old_extract_kernels_from_resnet_architecture(
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


def old_extract_kernels_from_inception_v3_architecture(
    model_children: list[nn.Module],
) -> tuple[list[torch.Tensor], list[nn.Conv2d]]:
    """
    Extracts the kernel weights and convolutional layers from an Inception V3
    architecture.

    Args:
        model_children: A list of child modules from the Inception V3 model.

    Returns:
        A tuple containing two lists:
            1. The weights of the extracted convolutional layers.
            2. The extracted convolutional layers themselves.
    """

    model_weights = []
    conv_layers = []
    # Initialize a counter to keep track of the number of convolutional layers
    counter = 0

    # Recursive function to search for Conv2d layers in nested modules
    def find_conv_layers(module: nn.Module):
        nonlocal counter

        for child in module.children():
            if isinstance(child, nn.Conv2d):
                counter += 1
                model_weights.append(child.weight)
                conv_layers.append(child)
            else:
                find_conv_layers(child)

    # Iterate through the model's child modules
    for i in range(len(model_children)):
        find_conv_layers(model_children[i])

    # Print the total number of convolutional layers found
    print(f"Total convolutional layers: {counter}")

    # Return the updated model_weights and conv_layers lists as a tuple
    return model_weights, conv_layers


def old_extract_feature_map_of_convolutional_layers(
        input_tensor: torch.Tensor,
        conv_layers: list[nn.Conv2d],
        model_name: str
    ) -> list[torch.Tensor]:
    """
    Extracts the feature maps of a list of convolutional layers applied to an input
    tensor.

    Args:
        input_tensor: The input tensor to pass through the convolutional layers.
        conv_layers: A list of convolutional layers to apply to the input tensor.
        model_name: The name of the model. Accepts 'resnet18' or 'inception_v3'.

    Returns:
        A list containing the feature maps of each convolutional layer applied to the
            input tensor.

    Raises:
        TypeError: If the input_tensor is not a torch.Tensor, or if conv_layers is not a
            list of nn.Conv2d layers.

    """

    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError("input_tensor must be a torch.Tensor")

    if not all(isinstance(layer, nn.Conv2d) for layer in conv_layers):
        raise TypeError("conv_layers must be a list of nn.Conv2d layers")

    if model_name not in ['resnet18', 'inception_v3']:
        raise ValueError("model_name must be either 'resnet18' or 'inception_v3'")

    results = [conv_layers[0](input_tensor)]

    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))

    return results





def old_visualize_feature_map_of_convolutional_layers(
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


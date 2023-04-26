import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchattacks
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import INCEPTIONV3_MODEL_NAME, RESNET18_MODEL_NAME


def train_and_evaluate_xgboost_classifier(
    benign_feature_map: list[list[float]] | np.ndarray,
    adversarial_feature_map: list[list[float]] | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[xgb.XGBClassifier, float]:
    """
    Trains an XGBoost classifier on input benign and adversarial feature maps, evaluates
    its accuracy, and returns the trained model and accuracy.

    Args:
        benign_feature_map (list of lists or array-like): A 2D list or array-like object
            containing the benign features.
        adversarial_feature_map (list of lists or array-like): A 2D list or array-like
            object containing the adversarial features.
        test_size (float, optional): A float between 0 and 1 representing the proportion
            of the dataset to be used as test set. Defaults to 0.2.
        random_state (int, optional): A random seed used by the random number generator.
            Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - model (xgb.XGBClassifier): The trained XGBoost classifier.
            - accuracy (float): The accuracy of the classifier on the test set, as a
                percentage.
    """
    X_train, X_test, y_train, y_test = prepare_data(
        benign_feature_map, adversarial_feature_map, test_size, random_state
    )
    model = train_xgboost_classifier(X_train, y_train)
    accuracy = evaluate_classifier(model, X_test, y_test)

    return model, accuracy

def train_xgboost_classifier(
    X_train: np.ndarray, y_train: np.ndarray
) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier on input training data and returns the trained model.

    Args:
        X_train (np.ndarray): A 2D numpy array containing the training features.
        y_train (np.ndarray): A 1D numpy array containing the training labels.

    Returns:
        xgb.XGBClassifier: The trained XGBoost classifier.
    """
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: xgb.XGBClassifier, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Evaluates the accuracy of the input classifier on the test set.

    Args:
        model (xgb.XGBClassifier): The trained classifier to be evaluated.
        X_test (np.ndarray): A 2D numpy array containing the test features.
        y_test (np.ndarray): A 1D numpy array containing the test labels.

    Returns:
        float: The accuracy of the classifier on the test set, as a percentage.
    """
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    return accuracy


def prepare_data(
    benign_feature_map: list[list[float]] | np.ndarray,
    adversarial_feature_map: list[list[float]] | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combines benign and adversarial feature maps, creates labels, and splits the data
    into training and testing sets.

    Args:
        benign_feature_map (list of lists or array-like): A 2D list or array-like object
            containing the benign features.
        adversarial_feature_map (list of lists or array-like): A 2D list or array-like
            object containing the adversarial features.
        test_size (float, optional): A float between 0 and 1 representing the proportion
            of the dataset to be used as test set. Defaults to 0.2.
        random_state (int, optional): A random seed used by the random number generator.
            Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - X_train (np.ndarray): A 2D numpy array containing the training features.
            - X_test (np.ndarray): A 2D numpy array containing the test features.
            - y_train (np.ndarray): A 1D numpy array containing the training labels.
            - y_test (np.ndarray): A 1D numpy array containing the test labels.
    """
    benign_features = np.array(benign_feature_map)
    adversarial_features = np.array(adversarial_feature_map)

    X = np.concatenate((benign_features, adversarial_features), axis=0)
    y = np.concatenate(
        (np.ones(len(benign_features)), np.zeros(len(adversarial_features))),
        axis=0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def generate_adversarial_input(
        input,
        label, 
        attack
        ) -> tuple[torch.tensor]:
    """
        Applies adversarial attack to the input, creating an adversarial example.

        Args:
            input: input image or tensor
            label: correct label, 1-dimensional format [0,1,0,...]
            attack: specific attack form torchattacks

        Returns:
            (adversarial_input): attack applied to input
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

    feature_map_after_attack = get_feature_maps(adversarial_input, model, model_name)

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
        # l_inf_distance = torch.norm(a - b, p=float('inf'), dim=1)

        # Compute the log distance
        # log_distance = torch.log(l_inf_distance)
        log_distance = torch.log(l2_distance)

        # will ensure that the values that are 0 are changed to 0 instead of inf/-inf
        finite_mask = torch.isfinite(log_distance)
        log_distance[~finite_mask] = 0  # Set non-real values to 0

        log_distances.append(log_distance.item())

    return log_distances


def get_normalized_values(data: list) -> list:
    # Normalize log_distances
    distances_tensor = torch.tensor(data)

    # Calculate the mean and standard deviation
    mean = torch.mean(distances_tensor)
    std = torch.std(distances_tensor)

    # Normalize the tensor around 1
    normalized_distances = (distances_tensor - mean) / std + 1

    # Convert the tensor back to a list if needed
    return normalized_distances.tolist()


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


def save_line_plots(log_distances, output_dir, file_name):
    # Create a dictionary to group data by name
    data_dict = {}
    for name, cur_distance in log_distances:
        if name not in data_dict:
            data_dict[name] = []
        data_dict[name].append(cur_distance)

    # Create a colormap
    cmap = plt.cm.get_cmap('viridis', len(data_dict))

    # Set the figure size (width, height) in inches
    plt.figure(figsize=(40, 25))

    # Plot the data with consistent colors for each name and only one label per name
    plotted_names = set()
    for i, (name, distances) in enumerate(data_dict.items()):
        label = name if name not in plotted_names else None
        plt.plot(distances, label=label, color=cmap(i))
        plotted_names.add(name)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Line Plots of Log Distances')
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


def save_average_line_plots(log_distances, output_dir, file_name):
    # Create a dictionary to group data by name
    data_dict = {}
    for name, cur_distance in log_distances:
        if name not in data_dict:
            data_dict[name] = []
        data_dict[name].append(cur_distance)

    # Create a colormap
    cmap = plt.cm.get_cmap('viridis', len(data_dict))

    # Set the figure size (width, height) in inches
    plt.figure(figsize=(12, 6))

    # Compute the average and variance, and plot as lines with shaded variance region
    x = np.arange(len(log_distances[0][1]))  # Assuming all lists have the same length
    plotted_names = set()
    for i, (name, distances) in enumerate(data_dict.items()):
        label = name if name not in plotted_names else None
        distances = np.array(distances)
        avg = np.mean(distances, axis=0)
        variance = np.var(distances, axis=0)
        std_dev = np.sqrt(variance)
        plt.plot(x, avg, label=label, color=cmap(i))
        plt.fill_between(x, avg - std_dev, avg + std_dev, color=cmap(i), alpha=0.3)
        plotted_names.add(name)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Average and Variance of Log Distances')
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


# ======================================================
# ================ PRIVATE FUNCTIONS ===================
# ======================================================
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
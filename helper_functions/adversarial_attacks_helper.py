import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from torchvision.models import ResNet, Inception3
import torch
import torchattacks
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from config import INCEPTIONV3_MODEL_NAME, RESNET18_MODEL_NAME
from torch.utils.data import DataLoader

def process_data_loader_and_generate_feature_maps(
        data_loader: DataLoader,
        adversarial_attack,
        model: torch.nn.Module,
        model_name: str,
        device: str,
        sample_limit: int = None
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Process the data from the data loader, generating benign and adversarial feature
    maps based on the specified adversarial attack.

    Args:
        data_loader (DataLoader): The data loader to process.
        adversarial_attack: The adversarial attack function to apply.
        model (torch.nn.Module): The model to use for generating feature maps.
        model_name (str): The name of the model being used.
        device (str): The device where tensors should be moved to ('cuda' or 'cpu').
        sample_limit (int, optional): The maximum number of samples to process.
            Defaults to None.

    Returns:
        tuple[list[list[float]], list[list[float]]]: A tuple containing two lists of
            lists of float values, where the first list corresponds to benign feature
            maps and the second list corresponds to adversarial feature maps.
    """
    benign_feature_map = []
    adversarial_feature_map = []

    for i, (input, true_label) in tqdm(enumerate(data_loader)):
        input = input.to(device)
        true_label = true_label.to(device)

        adv_input = generate_adversarial_input(input, true_label, adversarial_attack)


        dense_layers_benign = _get_dense_layers_resnet18(input, model)
        dense_layers_adversarial = _get_dense_layers_resnet18(input, model)
        print(dense_layers_benign)
        print()
        print(dense_layers_adversarial)


        benign_feature_map.append([
            tensor.mean().item()
            for tensor in get_feature_maps(input, model, model_name)
        ])
        adversarial_feature_map.append([
            tensor.mean().item()
            for tensor in get_feature_maps(adv_input, model, model_name)
        ])

        if sample_limit is not None and i >= sample_limit:
            break

    return benign_feature_map, adversarial_feature_map


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
    accuracy = evaluate_classifier_accuracy(model, X_test, y_test)

    return model, accuracy


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

    input_data = np.concatenate((benign_features, adversarial_features), axis=0)
    labels = np.concatenate(
        (np.ones(len(benign_features)), np.zeros(len(adversarial_features))),
        axis=0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        input_data, labels, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def train_xgboost_classifier(
    train_input: np.ndarray,
    train_labels: np.ndarray
) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier on input training data and returns the trained model.

    Args:
        train_input (np.ndarray): A 2D numpy array containing the training features.
        train_labels (np.ndarray): A 1D numpy array containing the training labels.

    Returns:
        xgb.XGBClassifier: The trained XGBoost classifier.
    """
    model = xgb.XGBClassifier(eval_metric="logloss")
    model.fit(train_input, train_labels)
    return model


def evaluate_classifier_accuracy(
    model: xgb.XGBClassifier,
    test_input: np.ndarray,
    test_labels: np.ndarray
) -> float:
    """
    Evaluates the accuracy of the input classifier on the test set.

    Args:
        model (xgb.XGBClassifier): The trained classifier to be evaluated.
        test_input (np.ndarray): A 2D numpy array containing the test features.
        test_labels (np.ndarray): A 1D numpy array containing the test labels.

    Returns:
        float: The accuracy of the classifier on the test set, as a percentage.
    """
    y_pred = model.predict(test_input)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy


def evaluate_classifier_metrics(
    model: xgb.XGBClassifier,
    test_input: np.ndarray,
    test_labels: np.ndarray
) -> tuple[int, int, int, int]:
    """
    Evaluates the classification metrics of the input classifier on the test set.

    Args:
        model (xgb.XGBClassifier): The trained classifier to be evaluated.
        test_input (np.ndarray): A 2D numpy array containing the test features.
        test_labels (np.ndarray): A 1D numpy array containing the test labels.

    Returns:
        tuple: A tuple containing true positive, true negative, false positive, and
            false negative values.
    """
    y_pred = model.predict(test_input)
    predictions = [round(value) for value in y_pred]
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()
    return tp, tn, fp, fn


def generate_adversarial_input(
    input: torch.Tensor,
    label: torch.Tensor,
    attack
) -> tuple[torch.tensor]:
    """
    Applies an adversarial attack to an input tensor and returns the adversarial
    example.

    Args:
        input (torch.Tensor): The input tensor to attack, with shape
            (batch_size, channels, height, width).
        label (torch.Tensor): The true label(s) for the input, with shape
            (batch_size, num_classes). Each row should be a one-hot vector, with a 1 in
            the position corresponding to the true class label and 0s elsewhere.
        attack: An instance of an attack class from the torchattacks library, e.g. FGSM,
            PGD, etc. This class should take the input and label tensors as arguments
            and return the corresponding adversarial example(s).

    Returns: The adversarial example(s) generated by the attack, with the same shape as
        the input tensor. If batch_size > 1, this will be a tensor of shape 
        (batch_size, channels, height, width).
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
def _get_feature_maps_inception_v3(input, model: Inception3):
    """Get feature maps from InceptionV3 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torchvision.models.Inception3): InceptionV3 model.

    Returns:
        List[torch.Tensor]: List of feature maps of shape (N, C, H, W).
    """
    feature_maps = []

    def hook(module, input, output):
        feature_maps.append(output.detach())
        print(f"Layer name: {module.__class__.__name__}")
        print(f"Tensor size: {output.size()}")
        print(f"Tensor:\n{output}")
    # List of InceptionV3 layers to extract feature maps from
    layers = [
        # model.Conv2d_1a_3x3,
        # model.Conv2d_2a_3x3,
        # model.Conv2d_2b_3x3,
        # model.Conv2d_3b_1x1,
        # model.Conv2d_4a_3x3,
        model.Mixed_5b.branch1x1,
        model.Mixed_5b.branch5x5_1,
        model.Mixed_5b.branch3x3dbl_1,
        model.Mixed_5b.branch3x3dbl_3,
        model.Mixed_5b.branch_pool,
        # model.Mixed_5c,
        # model.Mixed_5d,
        # model.Mixed_6a,
        # model.Mixed_6b,
        # model.Mixed_6c,
        # model.Mixed_6d,
        # model.Mixed_6e,
        # model.Mixed_7a,
        # model.Mixed_7b,
        # model.Mixed_7c
    ]

    # Register hook on each layer
    handles = [layer.register_forward_hook(hook) for layer in layers]

    _ = model(input)

    # Remove hook from each layer
    for handle in handles:
        handle.remove()

    sys.exit()
    return feature_maps


def _get_feature_maps_resnet18(input, model: ResNet):
    """Get feature maps from ResNet18 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torchvision.models.ResNet): ResNet18 model.

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


def _get_dense_layers_resnet18(input, model: ResNet):
    """Get dense layers from ResNet18 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torchvision.models.ResNet): ResNet18 model.

    Returns:
        List[torch.Tensor]: List of dense layer outputs.
    """
    dense_layers_output = []
    def hook(module, input, output):
        dense_layers_output.append(output.detach())

    # List of ResNet18 layers to extract dense layers from
    layers = [
        module for _, module in model.named_children()
        if isinstance(module, torch.nn.Linear)
    ]
    # Register hook on each layer
    handles = [layer.register_forward_hook(hook) for layer in layers]

    _ = model(input)

    # Remove hook from each layer
    for handle in handles:
        handle.remove()

    return dense_layers_output


def _get_dense_layers_inception_v3(input, model: Inception3):
    """Get dense layers from Inception V3 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torchvision.models.Inception3): Inception V3 model.

    Returns:
        List[torch.Tensor]: List of dense layer outputs.
    """
    dense_layers_output = []
    def hook(module, input, output):
        dense_layers_output.append(output.detach())

    # List of Inception V3 layers to extract dense layers from
    layers = [
        module for _, module in model.named_children()
        if isinstance(module, torch.nn.Linear)
    ]
    # Register hook on each layer
    handles = [layer.register_forward_hook(hook) for layer in layers]

    _ = model(input)

    # Remove hook from each layer
    for handle in handles:
        handle.remove()

    return dense_layers_output

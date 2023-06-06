import csv
import os
import time
from typing import Callable
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
from torchvision.utils import save_image
from data_classes import ActivationResults, Metric, ProcessResults, XGBoostClassifierResults


def process_and_extract_components_and_metrics(
        data_loader: DataLoader,
        adversarial_attack,
        model: torch.nn.Module,
        model_name: str,
        device: str,
        attack_name: str,
        sample_limit: int = None,
        include_dense_layers: bool = False
) -> ProcessResults:
    
    def update_feature_maps(
            feature_maps: Metric, input_data, model, model_name, metrics, before_activation):
        for metric_name, metric_fn in metrics.items():
            feature_map = _get_feature_map_apply_metric_fn(
                input_data, model, model_name, metric_fn, before_activation)
            getattr(feature_maps, metric_name).append(feature_map)

    metrics = {
        # 'mean': mean_metric,
        # 'l1': l1_distance_metric,
        'l2': l2_distance_metric,
        'linf': linfinity_distance_metric,
        # 'std': std_dev_metric,
        # 'var': variance_metric,
        # 'fro': frobenius_norm_metric,
    }

    benign_feature_maps_before = Metric([], [], [], [], [], [], [])
    adv_feature_maps_before = Metric([], [], [], [], [], [], [])
    benign_feature_maps_after = Metric([], [], [], [], [], [], [])
    adv_feature_maps_after = Metric([], [], [], [], [], [], [])
    benign_dense_layers = []
    adv_dense_layers = []
    correct = 0
    fooled = 0
    elapsed_times = []
    times = []

    for i, (input, true_label) in tqdm(enumerate(data_loader)):
        input = input.to(device)
        true_label = true_label.to(device)


        adv_input, elapsed_time = generate_adversarial_input(
            input, true_label, adversarial_attack, attack_name)
        
        save_image(input, f"{attack_name}_benign_input.png")
        save_image(adv_input, f"{attack_name}_adversarial_input.png")
        save_image(input-adv_input, f"{attack_name}_difference.png")
        save_image_normalize(
            adv_input-input, f"{attack_name}_difference.png", normalize=True)
        predicted_label = model(input)
        predicted_adversarial_label = model(adv_input)
        print("True Label, Predicted Label and Predicted Adversarial Label",
            np.argmax(true_label.detach().cpu().numpy()),
            np.argmax(predicted_label.detach().cpu().numpy()),
            np.argmax(predicted_adversarial_label.detach().cpu().numpy()))
        
        # used for evaluting the average time to generate adversarial attack
        elapsed_times.append(elapsed_time)
        
        correct, fooled = assess_attack_single_input(
            model, device, input, adv_input, true_label, (correct, fooled)
        )
        
        start_time = time.time()
        update_feature_maps(
            benign_feature_maps_before, input, model, model_name, metrics,
            before_activation=True)
        update_feature_maps(
            adv_feature_maps_before, adv_input, model, model_name, metrics,
            before_activation=True)
        update_feature_maps(
            benign_feature_maps_after, input, model, model_name, metrics,
            before_activation=False)
        update_feature_maps(
            adv_feature_maps_after, adv_input, model, model_name, metrics,
            before_activation=False)

        if include_dense_layers:
            benign_dense_layers.append(get_dense_layers(input, model, model_name))
            adv_dense_layers.append(get_dense_layers(adv_input, model, model_name))
        
        times.append(time.time()-start_time)
        if sample_limit is not None and i+1 >= sample_limit:
            break
    
    avg_time = sum(times) / len(times)
    print(f"Average time for extraction of features are {avg_time} {len(times)} iterations.")

    if correct == 0:
        return 0.0
    fooling_rate = fooled / correct

    avg_elapsed_time = sum(elapsed_times)/len(elapsed_times)
    print(f"Average time over {len(elapsed_times)} runs: {avg_elapsed_time} seconds")


    results = ProcessResults(
        before_activation=ActivationResults(
            benign_feature_maps=benign_feature_maps_before,
            adv_feature_maps=adv_feature_maps_before
        ),
        after_activation=ActivationResults(
            benign_feature_maps=benign_feature_maps_after,
            adv_feature_maps=adv_feature_maps_after
        ),
        fooling_rate=fooling_rate,
        benign_dense_layers=benign_dense_layers,
        adv_dense_layers=adv_dense_layers
    )

    return results






def train_and_evaluate_xgboost_classifier(
    benign_list: list[list[float]] | np.ndarray,
    adv_list: list[list[float]] | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> XGBoostClassifierResults:

    """
    Trains an XGBoost classifier on input benign and adversarial feature maps, evaluates
    its performance using confusion matrix, and returns the trained model along with the
    performance metrics.

    Args:
        benign_list (list of lists or array-like): A 2D list or array-like object
            containing the benign features.
        adv_list (list of lists or array-like): A 2D list or array-like
            object containing the adversarial features.
        test_size (float, optional): A float between 0 and 1 representing the proportion
            of the dataset to be used as test set. Defaults to 0.2.
        random_state (int, optional): A random seed used by the random number generator.
            Defaults to 42.

    Returns:
        An instance of XGBoostClassifierResults dataclass
    """
    start_time = time.time()
    X_train, X_test, y_train, y_test = prepare_data(
        benign_list, adv_list, test_size, random_state
    )
    

    model = train_xgboost_classifier(X_train, y_train)
    print("Train time: ", time.time()-start_time)
    accuracy = evaluate_classifier_accuracy(model, X_test, y_test)
    
    tp, tn, fp, fn = evaluate_classifier_metrics(model, X_test, y_test)

    # Get feature importances.
    feature_importances = model.feature_importances_.tolist()
    print(feature_importances)
    # Sort the feature importances in descending order and get the indices.
    indices = np.argsort(feature_importances)[::-1]
    print(indices)

    return XGBoostClassifierResults(
        model=model,
        accuracy=accuracy,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn
    )


def prepare_data(
    benign_list: list[list[float]] | np.ndarray,
    adv_list: list[list[float]] | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combines benign and adversarial feature maps, creates labels, and splits the data
    into training and testing sets.

    Args:
        benign_list (list of lists or array-like): A 2D list or array-like object
            containing the benign features.
        adv_list (list of lists or array-like): A 2D list or array-like
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
    benign_array = np.array(benign_list)
    adv_array = np.array(adv_list)

    input_data = np.concatenate((benign_array, adv_array), axis=0)
    labels = np.concatenate(
        (np.ones(len(benign_array)), np.zeros(len(adv_array))),
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
    attack,
    attack_name: str
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
    
    start_time = time.time()
    # Move inputs and labels to the specified device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    label = label.to(device)    
    # turns 1-dimensional list into 0-dimensional scalar, needed for attack
    label_argmax = torch.argmax(label, 1)

    adversarial_input = attack(input, label_argmax)

    end_time = time.time()
    elapsed_time = end_time - start_time
    return adversarial_input, elapsed_time


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


def assess_attack(
    model: torch.nn.Module,
    device: torch.device,
    inputs: list[torch.Tensor],
    adv_inputs: list[torch.Tensor],
    true_labels: list[torch.Tensor],
) -> float:
    """
    Calculates the fooling rate for images that are correctly labeled by the model.

    Args:
        model (torch.nn.Module): The trained model for evaluation.
        device (torch.device): The device (CPU or GPU) to perform the calculations on.
        inputs (list[torch.Tensor]): A list of input tensors.
        adv_inputs (list[torch.Tensor]): A list of adversarial input tensors.
        true_labels (list[torch.Tensor]): A list of true label tensors.

    Returns:
        float: The fooling rate for images that are correctly labeled by the model.
    """

    correct = 0
    fooled = 0

    for input, adv_input, true_label in zip(inputs, adv_inputs, true_labels):
        input = input.to(device)
        true_label = true_label.to(device)

        predicted_label = model(input)
        predicted_adversarial_label = model(adv_input)

        pred_label = np.argmax(predicted_label.detach().cpu().numpy())
        true_label_val = np.argmax(true_label.detach().cpu().numpy())
        pred_adv_label = np.argmax(predicted_adversarial_label.detach().cpu().numpy())

        if pred_label == true_label_val:
            correct += 1
            if pred_adv_label != true_label_val:
                fooled += 1

    if correct == 0:
        return 0.0

    fooling_rate = fooled / correct
    return fooling_rate


def assess_attack_single_input(
    model: torch.nn.Module,
    device: torch.device,
    input: torch.Tensor,
    adv_input: torch.Tensor,
    true_label: torch.Tensor,
    current_counters: tuple[int, int],
) -> tuple[int, int]:
    """
    Updates the fooling rate counters for a single input that is correctly labeled by
    the model.

    Args:
        model (torch.nn.Module): The trained model for evaluation.
        device (torch.device): The device (CPU or GPU) to perform the calculations on.
        input (torch.Tensor): An input tensor.
        adv_input (torch.Tensor): An adversarial input tensor.
        true_label (torch.Tensor): A true label tensor.
        current_counters (tuple[int, int]): A tuple containing the current counter
            values (correct, fooled).

    Returns:
        tuple[int, int]: The updated counter values (correct, fooled).
    """

    correct, fooled = current_counters

    input = input.to(device)
    adv_input = adv_input.to(device)
    true_label = true_label.to(device)

    predicted_label = model(input)
    predicted_adversarial_label = model(adv_input)

    pred_label = np.argmax(predicted_label.detach().cpu().numpy())
    true_label_val = np.argmax(true_label.detach().cpu().numpy())
    pred_adv_label = np.argmax(predicted_adversarial_label.detach().cpu().numpy())

    if pred_label == true_label_val:
        correct += 1
        if pred_adv_label != true_label_val:
            fooled += 1

    return correct, fooled


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


def mean_metric(tensor: torch.Tensor) -> float:
    return tensor.mean().item()

def l1_distance_metric(tensor: torch.Tensor) -> float:
    return torch.norm(tensor, p=1).item()

def l2_distance_metric(tensor: torch.Tensor) -> float:
    return torch.norm(tensor, p=2).item()

def linfinity_distance_metric(tensor: torch.Tensor) -> float:
    return torch.norm(tensor, p=float('inf')).item()

def std_dev_metric(tensor: torch.Tensor) -> float:
    return torch.std(tensor).item()

def variance_metric(tensor: torch.Tensor) -> float:
    return torch.var(tensor).item()

def frobenius_norm_metric(tensor: torch.Tensor) -> float:
    return torch.norm(tensor, p='fro').item()


def get_feature_maps(input, model, model_name, before_activation_fn):
    """Get feature maps from a given model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torch.nn.Module): Model to extract feature maps from.
        model_name (str): Name of the model to extract feature maps from.

    Returns:
        List[torch.Tensor]: List of feature maps of shape (N, C, H, W).
    """
    if model_name == INCEPTIONV3_MODEL_NAME:
        return _get_feature_maps_inception_v3(input, model, before_activation_fn)
    elif model_name == RESNET18_MODEL_NAME:
        return _get_feature_maps_resnet18(input, model, before_activation_fn)
    else:
        raise Exception("Not valid model name")

def get_dense_layers(input, model, model_name: str):
    if model_name == INCEPTIONV3_MODEL_NAME:
        return _get_dense_layers_inception_v3(input, model)
    elif model_name == RESNET18_MODEL_NAME:
        return _get_dense_layers_resnet18(input, model)
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


def extend_lists(*args: list[list[float]] | np.ndarray) -> list[list[float]]:
    """
    Concatenates the elements of multiple lists of lists or 2D arrays element-wise.

    Args:
        *args (list of lists or 2D array): A variable number of 2D list or array-like
            objects.

    Returns:
        list: A new list of lists where each element is the concatenation of the
            corresponding elements from each of the input lists.
    """
    if len(args) < 2:
        raise ValueError("At least two input lists are required.")
    
    list_lengths = [len(lst) for lst in args]
    if not all(l == list_lengths[0] for l in list_lengths):
        raise ValueError("All input lists must have the same length.")
    
    extended_list = [
        [elem for sublist in [lst[i] for lst in args] for elem in sublist]
        for i in range(list_lengths[0])
    ]
    return extended_list


def select_attack(model, attack_name):
    eps = 8/255
    attack_name = attack_name.lower()

    if attack_name == 'fgsm':
        return torchattacks.FGSM(model, eps=eps)
    elif attack_name == 'ffgsm':
        return torchattacks.FFGSM(model, eps=eps)
    elif attack_name == 'bim':
        return torchattacks.BIM(model, eps=eps)
    elif attack_name == 'cw':
        return torchattacks.CW(model)
    elif attack_name == 'deepfool':
        return torchattacks.DeepFool(model)
    elif attack_name == 'pgd':
        return torchattacks.PGD(model)
    elif attack_name == 'pgdl2':
        return torchattacks.PGDL2(model)
    elif attack_name == 'autoattack':
        return torchattacks.AutoAttack(model)
    else:
        raise ValueError(f"Unsupported attack name: {attack_name}")


def print_result(attack_title, acc, tp, tn, fp, fn):
    
        print(f"\n{attack_title}: {tp} & {tn} & {fp} & {fn} & {acc:.2f}\% ")

def print_results(*results_list):
    labels = [
        "Feature map mean",
        "Feature map L1",
        "Feature map L2",
        "Feature map Linf",
        # "Feature map std",
        # "Feature map var",
        # "Feature map fro",
        "Activations mean",
        "Activations L1",
        "Activations L2",
        "Activations Linf",
        # "Activations std",
        # "Activations var",
        # "Activations fro",
        "Dense layers"
    ]

    for label, result in zip(labels, results_list):
        print_result(
            label, result.accuracy * 100.0,
            result.tp, result.tn,
            result.fp, result.fn
        )

def evaluate_attack_metrics(results: ProcessResults):
    fooling_rate = results.fooling_rate * 100.0
    print("Fooling rate: %.2f%%" % fooling_rate)

    before_activation = results.before_activation
    after_activation = results.after_activation

    def train_and_evaluate(feature_map_1, feature_map_2):
        return train_and_evaluate_xgboost_classifier(feature_map_1, feature_map_2)

    metrics = {
        'feature_map_mean': train_and_evaluate(
            before_activation.benign_feature_maps.mean,
            before_activation.adv_feature_maps.mean),
        'feature_map_l1': train_and_evaluate(
            before_activation.benign_feature_maps.l1, 
            before_activation.adv_feature_maps.l1),
        'feature_map_l2': train_and_evaluate(
            before_activation.benign_feature_maps.l2, 
            before_activation.adv_feature_maps.l2),
        'feature_map_linf': train_and_evaluate(
            before_activation.benign_feature_maps.linf,
            before_activation.adv_feature_maps.linf),
        'feature_map_std': train_and_evaluate(
            before_activation.benign_feature_maps.std,
            before_activation.adv_feature_maps.std),
        'feature_map_var': train_and_evaluate(
            before_activation.benign_feature_maps.var,
            before_activation.adv_feature_maps.var),
        'feature_map_fro': train_and_evaluate(
            before_activation.benign_feature_maps.fro,
            before_activation.adv_feature_maps.fro),
        'activations_mean': train_and_evaluate(
            after_activation.benign_feature_maps.mean,
            after_activation.adv_feature_maps.mean),
        'activations_l1': train_and_evaluate(
            after_activation.benign_feature_maps.l1,
            after_activation.adv_feature_maps.l1),
        'activations_l2': train_and_evaluate(
            after_activation.benign_feature_maps.l2,
            after_activation.adv_feature_maps.l2),
        'activations_linf': train_and_evaluate(
            after_activation.benign_feature_maps.linf,
            after_activation.adv_feature_maps.linf),
        'activations_std': train_and_evaluate(
            after_activation.benign_feature_maps.std,
            after_activation.adv_feature_maps.std),
        'activations_var': train_and_evaluate(
            after_activation.benign_feature_maps.var,
            after_activation.adv_feature_maps.var),
        'activations_fro': train_and_evaluate(
            after_activation.benign_feature_maps.fro,
            after_activation.adv_feature_maps.fro),
        'dense_layers': train_and_evaluate(
            results.benign_dense_layers, results.adv_dense_layers)
    }
    print_results(
        metrics["feature_map_mean"],
        metrics["feature_map_l2"],
        metrics["feature_map_l1"],
        metrics["feature_map_linf"],
        # metrics["feature_map_std"],
        # metrics["feature_map_var"],
        # metrics["feature_map_fro"],
        metrics["activations_mean"],
        metrics["activations_l1"],
        metrics["activations_l2"],
        metrics["activations_linf"],
        # metrics["activations_std"],
        # metrics["activations_var"],
        # metrics["activations_fro"],
        metrics["dense_layers"]
    )

    return metrics


def write_results_to_csv(filename, model_name, attack_name, samples, accuracy):
    # Create or append to the CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Check if the file is empty, if so write the headers
        if file.tell() == 0:
            writer.writerow(['model_name', 'attack_name', 'samples', 'accuracy'])

        # Write the data
        writer.writerow([model_name, attack_name, samples, accuracy])


# ======================================================
# ================ PRIVATE FUNCTIONS ===================
# ======================================================
def _get_feature_maps_inception_v3(
        input,
        model: Inception3,
        before_activation_fn: bool
):
    """Get feature maps from InceptionV3 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torchvision.models.Inception3): InceptionV3 model.

    Returns:
        List[torch.Tensor]: List of feature maps of shape (N, C, H, W).
    """
    feature_maps = []

    def hook(module, input, output):
        if before_activation_fn:
            feature_maps.append(input[0].detach())
        else:
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
        model.Mixed_7c, # 16
    ]
    # Register hook on each layer
    handles = [layer.register_forward_hook(hook) for layer in layers]

    _ = model(input)

    # Remove hook from each layer
    for handle in handles:
        handle.remove()

    return feature_maps


def _get_feature_maps_resnet18(input, model: ResNet, before_activation_fn: bool):
    """Get feature maps from ResNet18 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (torchvision.models.ResNet): ResNet18 model.

    Returns:
        List[torch.Tensor]: List of feature maps of shape (N, C, H, W).
    """
    feature_maps = []
    def hook(module, input, output):
        if before_activation_fn:
            feature_maps.append(input[0].detach())
        else:
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

def _get_feature_maps_resnet18_new(
        input,
        model: ResNet,
        before_activation_fn: bool,
) -> list[torch.Tensor]:
    """Get feature maps from ResNet18 model.

    Args:
        input (torch.Tensor): Input tensor of shape (N, C, H, W).
        model (ResNet): ResNet18 model.

    Returns:
        list[torch.Tensor]: List of feature maps of shape (N, C, H, W).
    """
    feature_maps = []

    def hook(module, input, output):
        if before_activation_fn:
            feature_maps.append(input[0].detach())
        else:
            feature_maps.append(output.detach())

    layers = [
        model.conv1,
        model.layer1[0].conv1,
        model.layer1[0].conv2,
        model.layer1[1].conv1,
        model.layer1[1].conv2,
        model.layer2[0].conv1,
        model.layer2[0].conv2,
        model.layer2[1].conv1,
        model.layer2[1].conv2,
        model.layer3[0].conv1,
        model.layer3[0].conv2,
        model.layer3[1].conv1,
        model.layer3[1].conv2,
        model.layer4[0].conv1,
        model.layer4[0].conv2,
        model.layer4[1].conv1,
        model.layer4[1].conv2,
    ]

    handles = [layer.register_forward_hook(hook) for layer in layers]

    _ = model(input)

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

    # Convert dense layer outputs to list of float values
    return [
        output.item() 
            for layer_output in dense_layers_output 
                for output in layer_output.view(-1)
    ]


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

    # Convert dense layer outputs to list of float values
    return [
        output.item() 
            for layer_output in dense_layers_output 
                for output in layer_output.view(-1)
    ]


def _get_feature_map_apply_metric_fn(
        input: torch.Tensor,
        model: torch.nn.Module,
        model_name: str, 
        metric_fn: Callable[[torch.Tensor], float],
        before_activation_fn: bool
    ) -> list[float]:
    """
    Compute the feature map using a metric function on the given input.

    Args:
        input (torch.Tensor): The input tensor.
        model (torch.nn.Module): The model to be used.
        model_name (str): The name of the model.
        metric_fn (Callable[[torch.Tensor], float]): The metric function to be used for
            calculations.

    Returns:
        list[float]: The computed feature map.
    """
    
    return [
        metric_fn(tensor)
        for tensor in get_feature_maps(input, model, model_name, before_activation_fn)
    ]




def save_image_normalize(adv_input, filename, normalize=False):
    # Assuming that the adversarial input is a 4D tensor
    # of shape [batch_size, channels, height, width]
    # If the tensor values are not between [0, 1], uncomment the following lines:
    if normalize:
        adv_input = adv_input.clone()  # Clone to avoid modifying the original tensor
        adv_input -= adv_input.min()  # Shift to positive
        adv_input /= adv_input.max()  # Normalize to [0, 1]
    save_image(adv_input, filename)

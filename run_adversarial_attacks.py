import argparse
import numpy as np
import torch
import torchattacks
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import multiprocessing as mp
from config import (
    INCEPTIONV3_MODEL_NAME, PREPROCESS_INCEPTIONV3, PREPROCESS_RESNET18,
    RANDOM_SEED, RESNET18_MODEL_NAME, TRAINED_INCEPTION_V3_MODEL_2018,
    TRAINED_INCEPTION_V3_MODEL_2019, TRAINED_RESNET18_MODEL_2018,
    TRAINED_RESNET18_MODEL_2019
)
from helper_functions.adversarial_attacks_helper import (
    generate_adversarial_input,
    get_feature_maps,
    train_and_evaluate_xgboost_classifier,
)
from helper_functions.misc_helper import get_trained_or_default_model
from helper_functions.train_model_helper import get_data_loaders_by_year


def _initialize_model(model_name: str, model_file_name: str) -> torch.nn.Module:
    """
    Initialize the model.

    Returns:
        torch.nn.Module: Trained or default model.
    """
    print("get trained or default model...")
    return get_trained_or_default_model(
        model_name,
        model_file_name=model_file_name
    )

def _initialize_data_loader_inception_v3(year:str, is_augmented_dataset:bool):
    return get_data_loaders_by_year(year, PREPROCESS_INCEPTIONV3, is_augmented_dataset)


def _initialize_data_loader_resnet18(year:str, is_augmented_dataset:bool):
    return get_data_loaders_by_year(year, PREPROCESS_RESNET18, is_augmented_dataset)

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

def _get_correct_model_file_name(model_name: str, year: str) -> str:
    if model_name == INCEPTIONV3_MODEL_NAME and year == "2019":
        return TRAINED_INCEPTION_V3_MODEL_2019
    elif model_name == INCEPTIONV3_MODEL_NAME and year == "2018":
        return TRAINED_INCEPTION_V3_MODEL_2018
    elif model_name == RESNET18_MODEL_NAME and year == "2019":
        return TRAINED_RESNET18_MODEL_2019
    elif model_name == RESNET18_MODEL_NAME and year == "2018":
        return TRAINED_RESNET18_MODEL_2018









def main(year, model_name, is_augmented):
    mp.freeze_support()
    mp.set_start_method('spawn')
    # Set the randomness seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize empty lists
    # log_distances: list = []
    # comparison_distances: list = []
    # correct_labels: list = []
    # predicted_labels: list = []
    # predicted_adversarial_labels: list = []
    benign_feature_map: list = []
    adversarial_feature_map: list = []
    attacks: list[tuple] = []
    model_file_name = _get_correct_model_file_name(model_name, year)

    model = _initialize_model(
        model_name,
        model_file_name=model_file_name
    )
    
    # Initialize setup
    if model_name == RESNET18_MODEL_NAME:
        # Initialize setup
        train_dl, val_dl, test_dl, _ = _initialize_data_loader_resnet18(
            year, is_augmented
        )
    elif model_name == INCEPTIONV3_MODEL_NAME:
        train_dl, val_dl, test_dl, _ = _initialize_data_loader_inception_v3(
            year, is_augmented
        )
        train_dl, val_dl, test_dl_2018, _ = _initialize_data_loader_inception_v3(
            "2018", is_augmented
        )
    else:
        raise Exception("Not a valid model name")

    device = _initialize_device()
    fgsm_attack = torchattacks.FGSM(model, eps=2/255)
    cw_attack = torchattacks.CW(model)
    #attacks.append(("deepfool",torchattacks.DeepFool(model)))
    # attacks.append(("one_pixel",torchattacks.OnePixel(model)))

    print("FGSM")
    for i, (input, true_label) in tqdm(enumerate(train_dl)):
        input = input.to(device)
        true_label = true_label.to(device)

        adv_input = generate_adversarial_input(input, true_label, fgsm_attack)
        
        benign_feature_map.append([
            tensor.mean().item() 
                for tensor in get_feature_maps(input, model, model_name)
        ])
        adversarial_feature_map.append([
            tensor.mean().item() 
                for tensor in get_feature_maps(adv_input, model, model_name)
        ])
        
        if i >= 10000:
            break
            
        
    
    xgboost_model, accuracy = train_and_evaluate_xgboost_classifier(
        benign_feature_map,
        adversarial_feature_map
    )
    test_benign_feature_map = []
    test_adversarial_feature_map = []
    for i, (input, true_label) in tqdm(enumerate(test_dl_2018)):
        input = input.to(device)
        true_label = true_label.to(device)

        adv_input = generate_adversarial_input(input, true_label, cw_attack)
        
        test_benign_feature_map.append([
            tensor.mean().item() 
                for tensor in get_feature_maps(input, model, model_name)
        ])
        test_adversarial_feature_map.append([
            tensor.mean().item() 
                for tensor in get_feature_maps(adv_input, model, model_name)
        ])
        
    # Assuming pca_1_list and pca_2_list are your input features
    benign_features = np.array(test_benign_feature_map)
    adversarial_features = np.array(test_adversarial_feature_map)

    # Combine the two lists into one and create the corresponding labels
    X = np.concatenate((benign_features, adversarial_features), axis=0)
    y = np.concatenate(
        (np.ones(len(benign_features)), np.zeros(len(adversarial_features))),
        axis=0
    )
    
    # Predict on the test set
    y_pred = xgboost_model.predict(X)
    predictions = [round(value) for value in y_pred]

    # Evaluate the accuracy
    accuracy = accuracy_score(y, predictions)
    print("Accuracy on test dataset: %.2f%%" % (accuracy * 100.0))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Model-name, dataset-year, possibility to use non-augmented dataset" 
            " and add multiple learning rates."
        )
    )
    # Add argument for the dataset year
    parser.add_argument(
        "--year",
        required=True,
        choices=["2018", "2019"],
        help="Dataset for which to perform training on (2018 or 2019)."
    )

    # Add argument for the model type
    parser.add_argument(
        "--model",
        required=True,
        choices=[INCEPTIONV3_MODEL_NAME, RESNET18_MODEL_NAME],
        help=(
            f"Model for which to perform training ({INCEPTIONV3_MODEL_NAME}"
            f" or {RESNET18_MODEL_NAME})"
        )
    )

    # Add argument for using augmented dataset
    parser.add_argument(
        "--is-augmented",
        action="store_true",
        help="Use augmented dataset if specified."
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.year,
        args.model,
        args.is_augmented
    )

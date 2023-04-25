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
)
import xgboost as xgb
from sklearn.model_selection import train_test_split
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

def _initialize_data_loader_inception_v3(year):
    return get_data_loaders_by_year(year, PREPROCESS_INCEPTIONV3, False)


def _initialize_data_loader_resnet18(year):
    return get_data_loaders_by_year(year, PREPROCESS_RESNET18, False)

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









def main(year, model_name):
    mp.freeze_support()
    mp.set_start_method('spawn')
    # Set the randomness seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Initialize empty lists
    log_distances: list = []
    comparison_distances: list = []
    correct_labels: list = []
    predicted_labels: list = []
    predicted_adversarial_labels: list = []
    pca_1_list: list = []
    pca_2_list: list = []
    attacks: list[tuple] = []
    model_file_name = _get_correct_model_file_name(model_name, year)

    model = _initialize_model(
        model_name,
        model_file_name=model_file_name
    )
    
    # Initialize setup
    if model_name == RESNET18_MODEL_NAME:
        # Initialize setup
        train_data_loader, *_ = _initialize_data_loader_resnet18(year)
    elif model_name == INCEPTIONV3_MODEL_NAME:
        train_data_loader, *_ = _initialize_data_loader_inception_v3(year)
    else:
        raise Exception("Not a valid model name")

    device = _initialize_device()
    attacks.append(("fgsm",torchattacks.FGSM(model, eps=2/255)))
    attacks.append(("cw",torchattacks.CW(model)))
    #attacks.append(("deepfool",torchattacks.DeepFool(model)))
    attacks.append(("one_pixel",torchattacks.OnePixel(model)))

    for name, attack in attacks:
        for i, (input, true_label) in tqdm(enumerate(train_data_loader)):
            input = input.to(device)
            true_label = true_label.to(device)

            # map1 = get_feature_maps(input, model, model_name)
            # map2 = get_feature_maps(last_input, model, model_name)
            # comparison_distance = calculate_log_distances(map1, map2)
            # comparison_distances.append((name, comparison_distance))
            # assessment_results = assess_attack_and_log_distances(
            #     model,
            #     device,
            #     input,
            #     true_label,
            #     attack,
            #     model_name
            # )
            # cur_distance, correct_label, predicted_label, adv_label = assessment_results
            # log_distances.append((name, cur_distance))
            # correct_labels.append(correct_label)
            # predicted_labels.append(predicted_label)
            # predicted_adversarial_labels.append(adv_label)

            print(name)
            adv_input = generate_adversarial_input(input, true_label, attack)
            
            pca_1_list.append(
                [tensor.mean().item() for tensor in get_feature_maps(input, model, model_name)])
            pca_2_list.append(
                [tensor.mean().item() for tensor in get_feature_maps(adv_input, model, model_name)])
            
            if i >= 5:
                break
            
        
    

    # Assuming pca_1_list and pca_2_list are your input features
    pca_1_list = np.array(pca_1_list)
    pca_2_list = np.array(pca_2_list)

    # Combine the two lists into one and create the corresponding labels
    X = np.concatenate((pca_1_list, pca_2_list), axis=0)
    y = np.concatenate((np.ones(len(pca_1_list)), np.zeros(len(pca_2_list))), axis=0)

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost classifier
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))  


    
    
    # save_line_plots(
    #     log_distances,
    #     "./test_images/",
    #     "l2_log_100_iter.png"
    # )
    # save_average_line_plots(
    #     log_distances,
    #     "./test_images/",
    #     "average_l2_log_100_iter.png"
    # )
    # save_line_plots(
    #     comparison_distances,
    #     "./test_images/",
    #     "comparison_l2_log_100_iter.png"
    # )
    # save_average_line_plots(
    #     comparison_distances,
    #     "./test_images/",
    #     "comparison_average_l2_log_000_iter.png"
    # )
    

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

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.year,
        args.model
    )

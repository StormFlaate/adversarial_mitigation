import argparse
import numpy as np
import torch
import torchattacks
import multiprocessing as mp
from config import (
    INCEPTIONV3_MODEL_NAME, PREPROCESS_INCEPTIONV3, PREPROCESS_RESNET18,
    RANDOM_SEED, RESNET18_MODEL_NAME, TRAINED_INCEPTION_V3_MODEL_2018,
    TRAINED_INCEPTION_V3_MODEL_2019, TRAINED_RESNET18_MODEL_2018,
    TRAINED_RESNET18_MODEL_2019
)
from helper_functions.adversarial_attacks_helper import (
    process_and_extract_components_and_metrics,
    train_and_evaluate_xgboost_classifier,
)
from helper_functions.misc_helper import get_trained_or_default_model
from helper_functions.train_model_helper import get_data_loaders_by_year


def _initialize_model(model_name: str, model_file_name: str) -> torch.nn.Module:
    """
    Initialize the model.

    Returns
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
        train_dl, val_dl, test_dl_2018, _ = _initialize_data_loader_resnet18(
            "2018", is_augmented
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
    # fgsm_attack = torchattacks.FGSM(model, eps=8/255)
    # ifgsm_attack = torchattacks.BIM(model, eps=8/255)
    # cw_attack = torchattacks.CW(model)
    # deepfool_attack = torchattacks.DeepFool(model)
    # pgd_linf_attack = torchattacks.PGD(model)
    # pgd_l2_attack = torchattacks.PGDL2(model)
    autoattack_attack = torchattacks.AutoAttack()
    

    train_process_output = process_and_extract_components_and_metrics(
        train_dl, autoattack_attack, model, model_name, device, sample_limit=100)
    
    print(len(train_process_output[0]))
    print(len(train_process_output[1]))
    print(len(train_process_output[0][0]))
    print(len(train_process_output[1][1]))
    xgboost_model_feature_map, acc_feature_map = train_and_evaluate_xgboost_classifier(
        train_process_output[0],
        train_process_output[1]
    )
    xgboost_model_dense_layers, acc_dense_layers = train_and_evaluate_xgboost_classifier(
        train_process_output[2],
        train_process_output[3]
    )

    print("xgboost_model_feature_map: %.2f%%" % (acc_feature_map * 100.0))
    print("xgboost_model_dense_layers: %.2f%%" % (acc_dense_layers * 100.0))



    # test_benign, test_adv = process_and_extract_components_and_metrics(
    #     test_dl_2018, deepfool_attack, model, model_name, device)

    # test_input, _, test_label, __ = prepare_data(
    #     test_benign,
    #     test_adv,
    #     test_size=0.05 
    # )

    # # Evaluate the accuracy
    # accuracy = evaluate_classifier_accuracy(
    #     xgboost_model_feature_map, test_input, test_label)
    # print("Accuracy: (xgboost_model_feature_map): %.2f%%" % (accuracy * 100.0))

    

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
    # Default to use the non-augmented dataset
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

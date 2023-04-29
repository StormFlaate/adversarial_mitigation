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
    assess_attack,
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









def main(year, model_name, is_augmented, samples):
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
    attack = torchattacks.FGSM(model, eps=8/255)
    # attack = torchattacks.FFGSM(model, eps=8/255)
    # attack = torchattacks.BIM(model, eps=8/255)
    # attack = torchattacks.CW(model)
    # attack = torchattacks.DeepFool(model)
    # attack = torchattacks.PGD(model)
    # attack = torchattacks.PGDL2(model)
    # attack = torchattacks.AutoAttack(model)
    

    *output_before_activation_fn, all_inputs, all_adv_inputs, all_true_labels = (
        process_and_extract_components_and_metrics(
            train_dl, attack, model, model_name, device, True, sample_limit=samples,
            include_dense_layers=True
        )
    )
    fooling_rate = assess_attack(
        model, device, all_inputs, all_adv_inputs, all_true_labels)
    print("Fooling rate: %.2f%%" % (fooling_rate * 100.0))

    *output_after_activation_fn, all_inputs, all_adv_inputs, all_true_labels = (
        process_and_extract_components_and_metrics(
            train_dl, attack, model, model_name, device, True, sample_limit=samples,
            include_dense_layers=False
        )
    )
    fooling_rate = assess_attack(
        model, device, all_inputs, all_adv_inputs, all_true_labels)
    print("Fooling rate: %.2f%%" % (fooling_rate * 100.0))

    # before activation
    model, acc_feature_map_mean, tp_fm_mean, tn_fm_mean, fp_fm_mean, fn_fm_mean = (
        train_and_evaluate_xgboost_classifier(
            output_before_activation_fn[0],
            output_before_activation_fn[1]
        )
    )
    model, acc_feature_map_l2, tp_fm_l2, tn_fm_l2, fp_fm_l2, fn_fm_l2 = (
        train_and_evaluate_xgboost_classifier(
            output_before_activation_fn[2],
            output_before_activation_fn[3]
        )
    )
    model, acc_feature_map_inf, tp_fm_inf, tn_fm_inf, fp_fm_inf, fn_fm_inf = (
        train_and_evaluate_xgboost_classifier(
            output_before_activation_fn[4],
            output_before_activation_fn[5]
        )
    )
    model, acc_dense_layers, tp_dl, tn_dl, fp_dl, fn_dl = (
        train_and_evaluate_xgboost_classifier(
            output_before_activation_fn[6],
            output_before_activation_fn[7]
        )
    )

    # after activation
    model, acc_activations_mean, tp_act_mean, tn_act_mean, fp_act_mean, fn_act_mean = (
        train_and_evaluate_xgboost_classifier(
            output_after_activation_fn[0],
            output_after_activation_fn[1]
        )
    )
    model, acc_activations_l2, tp_act_l2, tn_act_l2, fp_act_l2, fn_act_l2 = (
        train_and_evaluate_xgboost_classifier(
            output_after_activation_fn[2],
            output_after_activation_fn[3]
        )
    )
    model, acc_activations_linf, tp_act_linf, tn_act_linf, fp_act_linf, fn_act_linf = (
        train_and_evaluate_xgboost_classifier(
            output_after_activation_fn[4],
            output_after_activation_fn[5]
        )
    )


    
    print("Feature map mean: %.2f%%" % (acc_feature_map_mean * 100.0))
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (tp_fm_mean, tn_fm_mean, fp_fm_mean, fn_fm_mean))
    print("Feature map L2: %.2f%%" % (acc_feature_map_l2 * 100.0))
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (tp_fm_l2, tn_fm_l2, fp_fm_l2, fn_fm_l2))
    print("Feature map Linf: %.2f%%" % (acc_feature_map_inf * 100.0))
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (tp_fm_inf, tn_fm_inf, fp_fm_inf, fn_fm_inf))
    print("Activations mean: %.2f%%" % (acc_activations_mean * 100.0))
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (tp_act_mean, tn_act_mean, fp_act_mean, fn_act_mean))
    print("Activations L2: %.2f%%" % (acc_activations_l2 * 100.0))
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (tp_act_l2, tn_act_l2, fp_act_l2, fn_act_l2))
    print("Activations Linf: %.2f%%" % (acc_activations_linf * 100.0))
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (tp_act_linf, tn_act_linf, fp_act_linf, fn_act_linf))
    print("Dense layers : %.2f%%" % (acc_dense_layers * 100.0))
    print("TP: %d, TN: %d, FP: %d, FN: %d" % (tp_dl, tn_dl, fp_dl, fn_dl))




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

    # Number of samples
    parser.add_argument(
        "--samples",
        required=True,
        type=int
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
        args.is_augmented,
        args.samples
    )

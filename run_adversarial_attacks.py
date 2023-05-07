import argparse
import numpy as np
import torch
from torch.utils.data import Subset, ConcatDataset
import multiprocessing as mp
from config import (
    INCEPTIONV3_MODEL_NAME, PREPROCESS_INCEPTIONV3, PREPROCESS_RESNET18,
    RANDOM_SEED, RESNET18_MODEL_NAME, TRAINED_INCEPTION_V3_MODEL_2018,
    TRAINED_INCEPTION_V3_MODEL_2019, TRAINED_RESNET18_MODEL_2018,
    TRAINED_RESNET18_MODEL_2019
)
from helper_functions.adversarial_attacks_helper import (
    evaluate_classifier_accuracy,
    extend_lists,
    prepare_data,
    print_result,
    process_and_extract_components_and_metrics,
    select_attack,
    train_and_evaluate_xgboost_classifier,
)
from helper_functions.misc_helper import get_trained_or_default_model
from helper_functions.train_model_helper import get_data_loader, get_data_loaders_by_year


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
    dataloader, _ =  get_data_loaders_by_year(
        year, PREPROCESS_INCEPTIONV3, is_augmented_dataset, split_dataset=False)
    return dataloader


def _initialize_data_loader_resnet18(year:str, is_augmented_dataset:bool):
    dataloader, _ = get_data_loaders_by_year(
        year, PREPROCESS_RESNET18, is_augmented_dataset, split_dataset=False)
    return dataloader
    


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









def main(year, model_name, is_augmented, samples, attack_name, all_attacks):
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
        dataloader = _initialize_data_loader_resnet18(
            year, is_augmented
        )
        
    elif model_name == INCEPTIONV3_MODEL_NAME:
        dataloader = _initialize_data_loader_inception_v3(
            year, is_augmented
        )
    else:
        raise Exception("Not a valid model name")

    device = _initialize_device()
    attacks: list = []

    if all_attacks:
        attacks.extend(["fgsm", "bim", "cw", "pgd"])
    else:
        attacks.append(attack_name)

    for attack_name in attacks:
        print(f"Attack: {attack_name}")
        attack = select_attack(model, attack_name)
        result = (
            process_and_extract_components_and_metrics(
                dataloader, attack, model, model_name, device, attack_name,
                sample_limit=samples, include_dense_layers=True
            )
        )
        print("Fooling rate: %.2f%%" % (result["fooling_rate"] * 100.0))


        _, acc_feature_map_mean, tp_fm_mean, tn_fm_mean, fp_fm_mean, fn_fm_mean = (
            train_and_evaluate_xgboost_classifier(
                result["before_activation"]["benign_feature_maps"]["mean"],
                result["before_activation"]["adv_feature_maps"]["mean"]
            )
        )
        _, acc_feature_map_l1, tp_fm_l1, tn_fm_l1, fp_fm_l1, fn_fm_l1 = (
            train_and_evaluate_xgboost_classifier(
                result["before_activation"]["benign_feature_maps"]["l1"],
                result["before_activation"]["adv_feature_maps"]["l1"]
            )
        )
        _, acc_feature_map_l2, tp_fm_l2, tn_fm_l2, fp_fm_l2, fn_fm_l2 = (
            train_and_evaluate_xgboost_classifier(
                result["before_activation"]["benign_feature_maps"]["l2"],
                result["before_activation"]["adv_feature_maps"]["l2"]
            )
        )
        _, acc_feature_map_linf, tp_fm_linf, tn_fm_linf, fp_fm_linf, fn_fm_linf = (
            train_and_evaluate_xgboost_classifier(
                result["before_activation"]["benign_feature_maps"]["linf"],
                result["before_activation"]["adv_feature_maps"]["linf"]
            )
        )

        _, acc_activations_mean, tp_act_mean, tn_act_mean, fp_act_mean, fn_act_mean = (
            train_and_evaluate_xgboost_classifier(
                result["after_activation"]["benign_feature_maps"]["mean"],
                result["after_activation"]["adv_feature_maps"]["mean"]
            )
        )

        _, acc_activations_l1, tp_act_l1, tn_act_l1, fp_act_l1, fn_act_l1 = (
            train_and_evaluate_xgboost_classifier(
                result["after_activation"]["benign_feature_maps"]["l1"],
                result["after_activation"]["adv_feature_maps"]["l1"]
            )
        )
        _, acc_activations_l2, tp_act_l2, tn_act_l2, fp_act_l2, fn_act_l2 = (
            train_and_evaluate_xgboost_classifier(
                result["after_activation"]["benign_feature_maps"]["l2"],
                result["after_activation"]["adv_feature_maps"]["l2"]
            )
        )
        _, acc_activations_linf, tp_act_linf, tn_act_linf, fp_act_linf, fn_act_linf = (
            train_and_evaluate_xgboost_classifier(
                result["after_activation"]["benign_feature_maps"]["linf"],
                result["after_activation"]["adv_feature_maps"]["linf"]
            )
        )

        _, acc_dense_layers, tp_dl, tn_dl, fp_dl, fn_dl = (
            train_and_evaluate_xgboost_classifier(
                result["benign_dense_layers"],
                result["adv_dense_layers"]
            )
        )


        _, acc_combination_dense_act_l2, tp_comb, tn_comb, fp_comb, fn_comb = (
            train_and_evaluate_xgboost_classifier(
                extend_lists(result["after_activation"]["benign_feature_maps"]["l2"],result["benign_dense_layers"]),
                extend_lists(result["after_activation"]["adv_feature_maps"]["l2"],result["adv_dense_layers"]),
            )
        )

        benign_combo_list = extend_lists(
            extend_lists(
                result["after_activation"]["benign_feature_maps"]["l2"],
                result["benign_dense_layers"]
            ),
            result["before_activation"]["benign_feature_maps"]["l2"]
        ),
        adv_combo_list = extend_lists(
            extend_lists(
                result["after_activation"]["adv_feature_maps"]["l2"],
                result["adv_dense_layers"]
            ),
            result["before_activation"]["benign_feature_maps"]["l2"]
        )

        print(len(benign_combo_list[0]))
        print(len(benign_combo_list[0][0]))
        print(len(adv_combo_list[0]))
        print(len(adv_combo_list[0][0]))

        combo_model, acc_combination_dense_act_l2_fm_linf, tp_comb_double, tn_comb_double, fp_comb_double, fn_comb_double = (
            train_and_evaluate_xgboost_classifier(
                benign_combo_list,
                adv_combo_list
            )
        )

        print_result(
            "Feature map mean", acc_feature_map_mean * 100.0,
            tp_fm_mean, tn_fm_mean, fp_fm_mean, fn_fm_mean
        )
        print_result(
            "Feature map L1", acc_feature_map_l1 * 100.0,
            tp_fm_l1, tn_fm_l1, fp_fm_l1, fn_fm_l1
        )
        print_result(
            "Feature map L2", acc_feature_map_l2 * 100.0,
            tp_fm_l2, tn_fm_l2, fp_fm_l2, fn_fm_l2
        )
        print_result(
            "Feature map Linf", acc_feature_map_linf * 100.0,
            tp_fm_linf, tn_fm_linf, fp_fm_linf, fn_fm_linf
        )
        print_result(
            "Activations mean", acc_activations_mean * 100.0,
            tp_act_mean, tn_act_mean, fp_act_mean, fn_act_mean
        )
        print_result(
            "Activations L1", acc_activations_l1*100.0,
            tp_act_l1, tn_act_l1, fp_act_l1, fn_act_l1
        )
        print_result(
            "Activations L2", acc_activations_l2*100.0,
            tp_act_l2, tn_act_l2, fp_act_l2, fn_act_l2
        )
        print_result(
            "Activations Linf", acc_activations_linf*100.0,
            tp_act_linf, tn_act_linf, fp_act_linf, fn_act_linf
        )
        print_result(
            "Dense layers", acc_dense_layers * 100.0, tp_dl, tn_dl, fp_dl, fn_dl
        )
        print_result(
            "Combination of dense layers and activations L2",
            acc_combination_dense_act_l2 * 100.0,
            tp_comb, tn_comb, fp_comb, fn_comb
        )
        print_result(
            "Combination of dense layers, activations L2 and feature maps Linf",
            acc_combination_dense_act_l2_fm_linf * 100.0,
            tp_comb_double, tn_comb_double, fp_comb_double, fn_comb_double
        )


        attack_name_transfer = "fgsm"
        attack_transfer = select_attack(model, "fgsm")
        result_transfer = (
            process_and_extract_components_and_metrics(
                dataloader, attack_transfer, model, model_name, device,
                attack_name_transfer, sample_limit=samples, include_dense_layers=True
            )
        )


        print("Fooling rate: %.2f%%" % (result_transfer["fooling_rate"] * 100.0))
        benign_list_transfer = extend_lists(
            extend_lists(
                result_transfer["after_activation"]["benign_feature_maps"]["l2"],
                result_transfer["benign_dense_layers"]
            ),
            result_transfer["before_activation"]["benign_feature_maps"]["l2"]
        )

        adv_list_transfer = extend_lists(
            extend_lists(
                result_transfer["after_activation"]["adv_feature_maps"]["l2"],
                result_transfer["adv_dense_layers"]
            ),
            result_transfer["before_activation"]["benign_feature_maps"]["l2"]
        )

        print(len(benign_list_transfer[0]))
        print(len(benign_list_transfer[0][0]))
        print(len(adv_list_transfer[0]))
        print(len(adv_list_transfer[0][0]))
        

        output = prepare_data(benign_list_transfer, adv_list_transfer)


        accuracy_transfer = evaluate_classifier_accuracy(
            combo_model, output[0], output[2])
        print(f"Transfer accuracy: {accuracy_transfer}")


    



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

    # Number of samples
    parser.add_argument(
        "--attack",
        required=True,
        type=str
    )


    # Add argument for using augmented dataset
    # Default to use the non-augmented dataset
    parser.add_argument(
        "--is-augmented",
        action="store_true",
        help="Use augmented dataset if specified."
    )

    parser.add_argument(
        "--all-attacks",
        action="store_true",
        help="Run all attacks to the specified model"
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(
        args.year,
        args.model,
        args.is_augmented,
        args.samples,
        args.attack,
        args.all_attacks
    )

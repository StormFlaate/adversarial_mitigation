import argparse
import time
import numpy as np
import torch
import multiprocessing as mp
from config import (
    INCEPTIONV3_MODEL_NAME, PREPROCESS_INCEPTIONV3, PREPROCESS_RESNET18,
    RANDOM_SEED, RESNET18_MODEL_NAME, TRAINED_INCEPTION_V3_MODEL_2018,
    TRAINED_INCEPTION_V3_MODEL_2019, TRAINED_RESNET18_MODEL_2018,
    TRAINED_RESNET18_MODEL_2019
)
from data_classes import XGBoostClassifierResults
from helper_functions.adversarial_attacks_helper import (
    evaluate_classifier_accuracy,
    evaluate_classifier_metrics,
    extend_lists,
    prepare_data,
    print_result,
    process_and_extract_components_and_metrics,
    select_attack,
    train_and_evaluate_xgboost_classifier,
    write_results_to_csv,
)
from helper_functions.misc_helper import get_trained_or_default_model
from helper_functions.train_model_helper import get_data_loaders_by_year, test_model


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
    *_, test_dataloader, _ =  get_data_loaders_by_year(
        year, PREPROCESS_INCEPTIONV3, is_augmented_dataset, split_dataset=True)
    return dataloader, test_dataloader


def _initialize_data_loader_resnet18(year:str, is_augmented_dataset:bool):
    dataloader, _ = get_data_loaders_by_year(
        year, PREPROCESS_RESNET18, is_augmented_dataset, split_dataset=False)
    
    *_, test_dataloader, _ = get_data_loaders_by_year(
        year, PREPROCESS_RESNET18, is_augmented_dataset, split_dataset=True)
    return dataloader, test_dataloader
    

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


def _initialize_model_wrapper(model_name, year):
    model_file_name = _get_correct_model_file_name(model_name, year)
    return _initialize_model(model_name, model_file_name=model_file_name)


def _initialize_dataloader(model_name, year, is_augmented):
    if model_name == RESNET18_MODEL_NAME:
        return _initialize_data_loader_resnet18(year, is_augmented)
    elif model_name == INCEPTIONV3_MODEL_NAME:
        return _initialize_data_loader_inception_v3(year, is_augmented)
    else:
        raise Exception("Not a valid model name")


def _get_attacks(all_attacks, attack_name):
    return ["fgsm", "bim", "cw", "pgd"] if all_attacks else [attack_name]


def _process_and_extract_metrics(
        dataloader, 
        attack, 
        model, 
        model_name, 
        device, 
        attack_name, 
        samples
):
    result = process_and_extract_components_and_metrics(
        dataloader, attack, model, model_name, device, attack_name, 
        sample_limit=samples, include_dense_layers=True)
    # evaluate_attack_metrics(result)
    return result



def _extend_lists(result, include_dense_layers):
    benign_list = extend_lists(
        result.after_activation.benign_feature_maps.l2, 
        result.before_activation.benign_feature_maps.linf)
    adv_list = extend_lists(
        result.after_activation.adv_feature_maps.l2, 
        result.before_activation.adv_feature_maps.linf)
    if include_dense_layers:
        benign_list = extend_lists(benign_list, result.benign_dense_layers)
        adv_list = extend_lists(adv_list, result.adv_dense_layers)
    return benign_list, adv_list


def _train_and_evaluate(benign_combo_list, adv_combo_list):
    return train_and_evaluate_xgboost_classifier(benign_combo_list, adv_combo_list)

def evaluate_xgboost_classifier(xgboost_results, benign_list, adv_list):
    output = prepare_data(benign_list, adv_list)

    transfer_accuracy = evaluate_classifier_accuracy(
                xgboost_results.model, output[0], output[2])
    transfer_confusion_matrix = evaluate_classifier_metrics(
                xgboost_results.model, output[0], output[2])

    print_result(
        "combo_l2_linf_dense",
        transfer_accuracy*100.0,
        *transfer_confusion_matrix
    )

def _evaluate_transfer_attack(
        result, model, model_name, device, attack_name, samples, dataloader,
        result_xgboost_1:XGBoostClassifierResults, 
        result_xgboost_2: XGBoostClassifierResults
    ) -> None:
    for attack_name_transfer in ["fgsm", "bim", "cw", "pgd"]:
        print("-"*50)
        print("Original attack: ", attack_name)
        print("Transfer attack: ", attack_name_transfer)

        attack_transfer = select_attack(model, attack_name_transfer)
        res_transfer = _process_and_extract_metrics(
            dataloader, attack_transfer, model, model_name, device, 
            attack_name_transfer, min(625, samples))
        
        print("Fooling rate: %.2f%%" % (res_transfer.fooling_rate * 100.0))
        benign_list_1, adv_list_1 = _extend_lists(
            res_transfer, include_dense_layers=False)
        evaluate_xgboost_classifier(result_xgboost_1, benign_list_1, adv_list_1)

        benign_list_2, adv_list_2 = _extend_lists(
            res_transfer, include_dense_layers=True)
        evaluate_xgboost_classifier(result_xgboost_2, benign_list_2, adv_list_2)
        print("-"*50)
        


def main(
        year, model_name, is_augmented, samples, attack_name, all_attacks, 
        evaluate_transfer):
    model = _initialize_model_wrapper(model_name, year)
    dataloader, test_dataloader = _initialize_dataloader(model_name, year, is_augmented)
    test_model(model, test_dataloader, model_name)

    device = _initialize_device()
    attacks = _get_attacks(all_attacks, attack_name)
    
    for attack_name in attacks:
        print("\n"+"#"*100)
        print(f"Attack: {attack_name}")
        start_time_attack = time.time()
        attack = select_attack(model, attack_name)
        result = _process_and_extract_metrics(
            dataloader, attack, model, model_name, device, attack_name, samples)
        
        # ===============================
        # ===== COMBINATIONS SINGLE =====
        # ===============================
        # benign_combo_list, adv_combo_list = _extend_lists(
        #     result, include_dense_layers=False)
        
        # result_xgboost_1: XGBoostClassifierResults = _train_and_evaluate(
        #     benign_combo_list, adv_combo_list)
        
        # print_result(
        #     "combo_l2_linf", result_xgboost_1.accuracy*100, result_xgboost_1.tp, 
        #     result_xgboost_1.tn, result_xgboost_1.fp, result_xgboost_1.fn)

        # ==============================
        # ===== COMBINATIONS DOUBLE ====
        # ==============================
        print("Attack time:",{time.time() - start_time_attack})
        benign_combo_list, adv_combo_list = _extend_lists(
            result, include_dense_layers=True)
        
        result_xgboost_2: XGBoostClassifierResults = _train_and_evaluate(
            benign_combo_list, adv_combo_list)
        
        print_result(
            "combo_l2_linf_dense", result_xgboost_2.accuracy*100, result_xgboost_2.tp,
            result_xgboost_2.tn, result_xgboost_2.fp, result_xgboost_2.fn)

        write_results_to_csv(
            "results.csv", model_name, attack_name, samples,
            result_xgboost_2.accuracy*100)

        # if evaluate_transfer:
        #     _evaluate_transfer_attack(
        #         result, model, model_name, device, attack_name, samples, dataloader,
        #         result_xgboost_1, result_xgboost_2)



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
        nargs='+',  # '+' means one or more arguments
        help="Dataset for which to perform training on (2018, 2019 or both)."
    )

    # Add argument for the model type
    parser.add_argument(
        "--model",
        required=True,
        nargs='+',  # '+' means one or more arguments
        help=(
            f"Model for which to perform training. Options are "
            f"{INCEPTIONV3_MODEL_NAME} and/or {RESNET18_MODEL_NAME}"
        )
    )

    parser.add_argument(
        "--samples",
        required=True,
        type=int
    )

    parser.add_argument(
        "--all-attacks",
        action="store_true",
        help="Run all attacks to the specified model"
    )

    parser.add_argument(
        "--attack",
        required=False,
        type=str
    )

    # Default to use the non-augmented dataset
    parser.add_argument(
        "--is-augmented",
        action="store_true",
        help="Use augmented dataset if specified."
    )

    parser.add_argument(
        "--evaluate-transfer",
        action="store_true",
        help="Evaluate how good the model performs on the other attacks."
    )


    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if --attack is provided when --all-attacks is False
    if not args.all_attacks and args.attack is None:
        parser.error("--attack is required if --all-attacks is not specified")

    # Call the main function with parsed arguments
    mp.freeze_support()
    mp.set_start_method('spawn')

    # Set the randomness seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for model_name in args.model:  # loop through user specified models
        for year in args.year:  # loop through user specified years
            print("Model name:", model_name)
            print("Year:", year)
            main(
                year, model_name, args.is_augmented, args.samples,
                args.attack, args.all_attacks, args.evaluate_transfer)


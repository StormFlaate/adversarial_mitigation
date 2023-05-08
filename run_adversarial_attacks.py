import argparse
import numpy as np
import torch
import multiprocessing as mp
from config import (
    INCEPTIONV3_MODEL_NAME, PREPROCESS_INCEPTIONV3, PREPROCESS_RESNET18,
    RANDOM_SEED, RESNET18_MODEL_NAME, TRAINED_INCEPTION_V3_MODEL_2018,
    TRAINED_INCEPTION_V3_MODEL_2019, TRAINED_RESNET18_MODEL_2018,
    TRAINED_RESNET18_MODEL_2019
)
from data_classes import ProcessResults
from helper_functions.adversarial_attacks_helper import (
    evaluate_attack_metrics,
    evaluate_classifier_accuracy,
    evaluate_classifier_metrics,
    extend_lists,
    prepare_data,
    print_result,
    process_and_extract_components_and_metrics,
    select_attack,
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
        print()
        print("#"*100)
        print(f"Attack: {attack_name}")
        attack = select_attack(model, attack_name)
        
        result: ProcessResults = (
        process_and_extract_components_and_metrics(
            dataloader, attack, model, model_name, device, attack_name,
            sample_limit=samples, include_dense_layers=True
            )
        )
        before_activation = result.before_activation
        after_activation = result.after_activation
        evaluate_attack_metrics(result)


        benign_combo_list = extend_lists(
            after_activation.benign_feature_maps.l2,
            before_activation.benign_feature_maps.linf)
        adv_combo_list = extend_lists(
            after_activation.adv_feature_maps.l2,
            before_activation.adv_feature_maps.linf)

        combo_l2_linf = train_and_evaluate_xgboost_classifier(
            benign_combo_list,
            adv_combo_list
        )

        print_result(
            "combo_l2_linf",
            combo_l2_linf.accuracy*100,
            combo_l2_linf.tp,
            combo_l2_linf.tn,
            combo_l2_linf.fp,
            combo_l2_linf.fn
        )

        benign_combo_list = extend_lists(
            after_activation.benign_feature_maps.l2,
            before_activation.benign_feature_maps.linf,
            result.benign_dense_layers)
        adv_combo_list = extend_lists(
            after_activation.adv_feature_maps.l2,
            before_activation.adv_feature_maps.linf,
            result.adv_dense_layers)

        combo_l2_linf_dense = train_and_evaluate_xgboost_classifier(
            benign_combo_list,
            adv_combo_list
        )

        print_result(
            "combo_l2_linf_dense",
            combo_l2_linf_dense.accuracy*100,
            combo_l2_linf_dense.tp,
            combo_l2_linf_dense.tn,
            combo_l2_linf_dense.fp,
            combo_l2_linf_dense.fn
        )

        for attack_name_transfer in ["fgsm", "bim", "cw", "pgd"]:
            print("-"*50)
            print("Original attack: ", attack_name)
            print("Transfer attack: ", attack_name_transfer)
            attack_transfer = select_attack(model, attack_name_transfer)
            res_transfer: ProcessResults = process_and_extract_components_and_metrics(
                dataloader, attack_transfer, model, model_name, device,
                attack_name_transfer, sample_limit=min(625, samples),
                include_dense_layers=True
            )
            print("Fooling rate: %.2f%%" % (res_transfer.fooling_rate * 100.0))

            benign_list_transfer_1 = extend_lists(
                res_transfer.after_activation.benign_feature_maps.l2,
                res_transfer.before_activation.benign_feature_maps.linf
            )
            adv_list_transfer_1 = extend_lists(
                res_transfer.after_activation.adv_feature_maps.l2,
                res_transfer.before_activation.adv_feature_maps.linf
            )
                    
            output_1 = prepare_data(benign_list_transfer_1, adv_list_transfer_1)

            transfer_accuracy = evaluate_classifier_accuracy(
                combo_l2_linf.model, output_1[0], output_1[2])
            transfer_confusion_matrix = evaluate_classifier_metrics(
                combo_l2_linf.model, output_1[0], output_1[2])

            print_result(
                "combo_l2_linf",
                transfer_accuracy*100.0,
                *transfer_confusion_matrix
            )


            # Double combo
            benign_list_transfer_2 = extend_lists(
                res_transfer.after_activation.benign_feature_maps.l2,
                res_transfer.before_activation.benign_feature_maps.linf,
                res_transfer.benign_dense_layers
            )
            adv_list_transfer_2 = extend_lists(
                res_transfer.after_activation.adv_feature_maps.l2,
                res_transfer.before_activation.adv_feature_maps.linf,
                res_transfer.adv_dense_layers
            )
                    
            output_2 = prepare_data(benign_list_transfer_2, adv_list_transfer_2)

            transfer_accuracy = evaluate_classifier_accuracy(
                combo_l2_linf_dense.model, output_2[0], output_2[2])
            transfer_confusion_matrix = evaluate_classifier_metrics(
                combo_l2_linf_dense.model, output_2[0], output_2[2])

            print_result(
                "combo_l2_linf_dense",
                transfer_accuracy*100.0,
                *transfer_confusion_matrix
            )

    

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
    # main(
    #     args.year,
    #     args.model,
    #     args.is_augmented,
    #     args.samples,
    #     args.attack,
    #     args.all_attacks
    # )
    mp.freeze_support()
    mp.set_start_method('spawn')
    # Set the randomness seeds
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for model_name in ["resnet18", "inception_v3"]:
        for year in ["2018", "2019"]:
            print("Model name:", model_name)
            print("Year:", year)
            main(year, model_name, False, args.samples, "bim", True)

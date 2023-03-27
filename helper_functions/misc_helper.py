import os
import torch
import uuid
import torch.hub
import torch.nn as nn
from datetime import date
from config import MODEL_NAME

def truncated_uuid4() -> str:
    """Generate a truncated UUID-4 string.

    Returns:
        A string containing the first 8 characters of a UUID-4.
    """
    return str(uuid.uuid4())[:8]


def get_trained_or_default_model(
        model_file_name: str = "test_model", 
        models_dir: str = "models", 
        model_name: str = MODEL_NAME
        ) -> nn.Module:
    """
    Load a PyTorch model from a file if it exists, otherwise load the pretrained default model.

    Args:
        model_file_name (str): The name of the model file to load.
        models_dir (str): The path to the directory containing the model file.
        model_name (str): The name of the pretrained default model to load.

    Returns:
        A PyTorch nn.Module representing the loaded model.
    """
    # Generate the file path for the model file
    filepath: str = _generate_filepath(model_file_name, models_dir)
    print(f"filepath: {filepath}")

    # Check if the model file exists
    if model_file_name == "test_model" or not file_exists(filepath):
        
        # Load the pretrained default model 
        print("Loading pretrained default model...")
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    else:
        # Load the model from the file
        print(f"Loading pretrained {model_file_name} model...")
        model = load_model_from_file(model_file_name, models_dir)

    # set mode to evaluation mode
    model.eval()
    return model

    
def save_model_and_parameters_to_file(
        model: torch.nn.Module, 
        model_file_name: str, 
        train_dataset_root_dir: str, 
        epoch: int,
        models_dir: str = "models",
        ) -> str:
    """
    Saves a PyTorch model to a file.

    Args:
        model: The PyTorch model to save.
        model_file_name: The name of the model to save.
        train_dataset_root_dir: The full path to the training dataset which was used for training
        models_dir: The directory to save the model file to.

    Returns:
        The filename of the saved model.
    """

    # Generate a unique ID to append to the model name
    model_id = str(uuid.uuid4().hex)[:3]
    
    print(train_dataset_root_dir)
    train_set_name: str = train_dataset_root_dir.replace("./", "").replace("/", "_")

    today = date.today()


    # Build the filename for the model file
    filename = f"{model_file_name}_{train_set_name}_{today}_{epoch}__{model_id}"
    filepath = f"{models_dir}/{filename}.pt"
    # saving the parameters to file
    _save_config_to_file(filename, models_dir)
    # Save the model to the file
    torch.save(model, filepath)

    return filename


def load_model_from_file(model_file_name: str, models_dir: str = "models") -> torch.nn.Module:
    """
    Loads a PyTorch model from a file.

    Args:
        model_file_name: The name of the model to load.
        models_dir: The directory containing the model file.

    Returns:
        The PyTorch model.
    """
    
    filepath: str = _generate_filepath(model_file_name, models_dir)
    
    # Load the model from the file
    model = torch.load(filepath)

    return model


def file_exists(file_path: str) -> bool:
    """
    Check if a file exists at the given path.

    Args:
        file_path: A string representing the file path to check.

    Returns:
        A boolean value indicating if the file exists or not.
    """
    return os.path.isfile(file_path)


def folder_exists(folder_path: str) -> bool:
    """
    Check if a folder exists at the given path.

    Args:
        folder_path: A string representing the folder path to check.

    Returns:
        A boolean value indicating if the folder exists or not.
    """
    return os.path.isdir(folder_path)


###################################################
# ============== PRIVATE FUNCTIONS ============== #
###################################################
def _save_config_to_file(filename: str, models_dir: str) -> None:
    """
    Saves the config.py file to a .txt file format, with the same name as the saved model

    Args:
        filename: The name of the file, which will be the same as the file for the model
        models_dir: The directory where the model and the configurations will be place

    Returns:
        None
    """
    with open("config.py") as f:
        data = f.read()
        f.close()

    filepath = f"{models_dir}/config_{filename}.txt"

    with open(filepath, mode="w") as f:
        f.write(data)
        f.close()

    
def _generate_filepath(model_file_name: str, models_dir: str) -> str:
    """Generate a file path for a given model file name in a models directory.

    Args:
        model_file_name (str): The name of the model file, with or without the '.pt' extension.
        models_dir (str): The path to the directory where model files are stored.

    Returns:
        str: The full file path for the model file, with '.pt' extension.

    """
    # Check if the file name already includes the '.pt' extension
    if model_file_name[-3:] == ".pt":
        # If yes, remove the extension
        model_file_name = model_file_name.replace(".pt", "")
    
    # Build the file path by concatenating the models directory path and the model file name with '.pt' extension
    filepath = f"{models_dir}/{model_file_name}.pt"

    return filepath

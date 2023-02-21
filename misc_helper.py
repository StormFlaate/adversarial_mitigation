import torch
import uuid
from datetime import date

def truncated_uuid4() -> str:
    """Generate a truncated UUID-4 string.

    Returns:
        A string containing the first 8 characters of a UUID-4.
    """
    return str(uuid.uuid4())[:8]


def save_model_and_parameters_to_file(
        model: torch.nn.Module, 
        model_name: str, 
        train_dataset_root_dir: str, 
        models_dir: str = "models"
        ) -> str:
    """
    Saves a PyTorch model to a file.

    Args:
        model: The PyTorch model to save.
        model_name: The name of the model to save.
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
    filename = f"{model_name}_{train_set_name}_{today}__{model_id}"
    filepath = f"{models_dir}/{filename}.pt"
    # saving the parameters to file
    _save_config_to_file(filename, models_dir)
    # Save the model to the file
    torch.save(model.state_dict(), filepath)

    return filename



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
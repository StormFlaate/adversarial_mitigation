import torch
import uuid

def truncated_uuid4() -> str:
    """Generate a truncated UUID-4 string.

    Returns:
        A string containing the first 8 characters of a UUID-4.
    """
    return str(uuid.uuid4())[:8]


def save_model_to_file(model: torch.nn.Module, model_name: str, training_set_name: str, models_dir: str = "models") -> str:
    """
    Saves a PyTorch model to a file.

    Args:
        model: The PyTorch model to save.
        model_name: The name of the model to save.
        training_set_name: The name of the training set, which was used to train the model
        models_dir: The directory to save the model file to.

    Returns:
        The filename of the saved model.
    """
    # Generate a unique ID to append to the model name
    model_id = str(uuid.uuid4().hex)[:6]

    # Build the filename for the model file
    filename = f"{model_name}-{training_set_name}-{model_id}.pt"
    filepath = f"{models_dir}/{filename}"

    # Save the model to the file
    torch.save(model.state_dict(), filepath)

    return filename

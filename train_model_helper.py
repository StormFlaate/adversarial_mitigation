from typing import Dict
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, Subset, random_split
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from customDataset import ISICDataset
from sklearn.metrics import accuracy_score, f1_score
from config import (
    IS_2018_DATASET, NUM_WORKERS, PIN_MEMORY_TRAIN_DATALOADER, PREPROCESS_TRANSFORM,
    SHUFFLE_TRAIN_DATALOADER, TRAIN_NROWS, BATCH_SIZE, TEST_NROWS,
    TEST_DATASET_LABELS, TEST_DATASET_ROOT_DIR, SHUFFLE_VAL_DATALOADER,
    TEST_SPLIT_PERCENTAGE, TRAIN_DATASET_LABELS, TRAIN_DATASET_ROOT_DIR,
    IMAGE_FILE_TYPE, TRAIN_SPLIT_PERCENTAGE, VAL_SPLIT_PERCENTAGE)
import warnings
import math
from typing import (
    List,
    Sequence,
    Union,
    TypeVar,
    Tuple
)
from torch import randperm
from torch._utils import _accumulate
from misc_helper import save_model_and_parameters_to_file
T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

def train_model(
        model: Module,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        criterion: Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model_name: str = "",
        epoch_count: int = 20,
        requires_grad: bool = True) -> Module:
    """Trains a neural network model by retraining a model with already existing weights
        form Image Net.

    Args:
        model: The neural network model to train.
        train_dataset: The train_dataset used for training.
        train_data_loader: The data loader used for iterating over the train_dataset.
        val_data_loader: The data loader used for validating the training
        criterion: The loss function used for calculating the loss between model output and labels.
        optimizer: The optimizer used for updating the model parameters.
        model_name: The type of the model being used (if applicable).
        epoch_count: The number of epochs to train the model.

    Returns:
        The trained neural network model.
    """
    
    # Freeze the model parameters to prevent backpropagation
    for param in model.parameters():
        param.requires_grad = requires_grad

    # Replace the final layer with a new layer that matches the number of classes in the train_dataset
    train_dataset = train_data_loader.dataset.dataset
    num_classes = len(train_dataset.annotations.columns)-1
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Is CUDA available:", torch.cuda.is_available())


    model.to(device)

    # Loop over the number of epochs
    for epoch in range(epoch_count):
        # Initialize the running loss for this epoch
        running_loss = 0.0
        # Loop over the data in the data loader
        for i, data in tqdm(enumerate(train_data_loader, 0)):
            # Get the inputs and labels from the data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert labels to a tensor of type float
            labels = torch.tensor(labels, dtype=torch.float).clone().detach()

            # If the model type is "inceptionv3", pass inputs through the model and get two outputs
            if model_name == "inceptionv3":
                outputs, x = model(inputs)
            # Otherwise, pass inputs through the model and get one output
            else: 
                outputs = model(inputs)
            
            # Calculate the loss between the model output and the labels
            loss = criterion(outputs, labels)
            
            # Backpropagate the loss through the model to calculate gradients
            loss.backward()
            
            # Update the model parameters using the gradients and the optimizer
            optimizer.step()
            
            # Add the loss for this batch to the running loss for this epoch
            running_loss += loss.item()
        
        # zero gradients after each epoch
        optimizer.zero_grad()
        scheduler.step()

        if epoch and epoch%10==0:
            # check the accuracy of the model
            overall_accuracy, overall_f1_score, accuracy_by_type_dict = _validate_model_during_training(model, val_data_loader)
            _print_test_results(overall_accuracy, overall_f1_score, accuracy_by_type_dict)
            
            # save the model to file
            save_model_and_parameters_to_file(model, model_name, TRAIN_DATASET_ROOT_DIR, epoch, models_dir="models")

        # Print the average loss for this epoch
        print('Epoch {} loss: {:.4f}'.format(epoch + 1, running_loss / (i + 1)))

    # Print a message indicating that training has finished
    print('Finished training')


    return model

def test_model(model: torch.nn.Module, data_loader: DataLoader[Subset[ISICDataset]], model_name: str="") -> None:
    """Tests a neural network model on a dataset and prints the accuracy and F1 score.

    Args:
        model: The neural network model to test.
        data_loader: The data loader used for iterating over the dataset.
    """
    # Put the model in evaluation mode to turn off dropout and batch normalization
    model.eval()

    # Define the list of target labels and predicted labels
    target_labels = []
    predicted_labels = []
    # extract the correct indecis from the dataset
    df: pd.DataFrame = _get_annotations_subset(data_loader) 

    # Initialize the dictionary of accuracy for each skin lesion type
    accuracy_by_type = {col: {"correct": 0, "total": 0} for col in df.columns[1:]}

    # Loop over the data in the data loader
    for i, data in tqdm(enumerate(data_loader, 0)):
        # Get the inputs and labels from the data
        inputs, labels = data

        # Move inputs and labels to the specified device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        labels = labels.to(device)

   
        outputs = model(inputs)

        # Convert the labels to a list of labels
        labels = torch.argmax(labels, 1)
        # Convert the predicted outputs to a list of labels
        predicted = torch.argmax(outputs.data, 1)


        np_labels = labels.cpu().numpy()
        np_predicted = predicted.cpu().numpy()

        # Append the target and predicted labels to their respective lists
        target_labels.extend(np_labels)
        predicted_labels.extend(np_predicted)


        # Calculate the accuracy for each skin lesion type
        for j in range(len(np_labels)):
            for k, col in enumerate(df.columns[1:]):
                if np_labels[j] == k:
                    accuracy_by_type[col]["total"] += 1
                    if np_predicted[j] == k:
                        accuracy_by_type[col]["correct"] += 1

    # Calculate the overall accuracy and F1 score
    overall_accuracy = accuracy_score(target_labels, predicted_labels)
    overall_f1_score = f1_score(target_labels, predicted_labels, average="weighted")

    # Print the overall accuracy and F1 score
    print("Overall accuracy: {:.4f}".format(overall_accuracy))
    print("Overall F1 score: {:.4f}".format(overall_f1_score))

    # Print the accuracy for each skin lesion type
    for col in df.columns[1:]:
        if accuracy_by_type[col]["total"]:
            accuracy = accuracy_by_type[col]["correct"] / accuracy_by_type[col]["total"]
        else:
            accuracy = 0
        print("{} accuracy: {:.4f}".format(col, accuracy))



def get_category_counts(data_loader: DataLoader[Subset[ISICDataset]]) -> Dict[str, int]:
    """Get the counts of each category in the dataset.

    Args:
        data_loader: The data loader for the dataset.

    Returns:
        A dictionary containing the name and count of each category.
    """

    df: pd.DataFrame = _get_annotations_subset(data_loader) 
    # Initialize the dictionary to store the counts
    count_dict = {}
    # Loop over the columns of the DataFrame
    for col in df.columns[1:]:
        # Count the number of occurrences of '1' in the column and add it to the dictionary
        count_dict[col] = df[col].sum()

    return count_dict
    

def random_split(dataset: Dataset[T], lengths: Sequence[Union[int, float]]) -> List[Subset[T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.


    >>> random_split(range(10), [3, 7])
    >>> random_split(range(30), [0.3, 0.3, 0.4])

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        print(sum(lengths), len(dataset))
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]




def get_data_loaders(
    train_dataset_labels: str=TRAIN_DATASET_LABELS,
    train_dataset_root_dir: str=TRAIN_DATASET_ROOT_DIR, 
    train_nrows: int=TRAIN_NROWS,
    test_dataset_labels: str=TEST_DATASET_LABELS, 
    test_dataset_root_dir: str=TEST_DATASET_ROOT_DIR, 
    test_nrows: int=TEST_NROWS,
    image_file_type: str=IMAGE_FILE_TYPE, 
    preprocess_transform=PREPROCESS_TRANSFORM,
    train_split_percentage: float=TRAIN_SPLIT_PERCENTAGE, 
    val_split_percentage: float=VAL_SPLIT_PERCENTAGE, 
    test_split_percentage: float=TEST_SPLIT_PERCENTAGE,
    batch_size: int=BATCH_SIZE, 
    shuffle_train_dataloader: bool=SHUFFLE_TRAIN_DATALOADER, 
    shuffle_val_dataloader: bool=SHUFFLE_VAL_DATALOADER,
    num_workers: int=NUM_WORKERS, 
    pin_memory_train_dataloader: bool=PIN_MEMORY_TRAIN_DATALOADER,
    is_2018_dataset:bool or None=IS_2018_DATASET
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    Returns train_data_loader, val_data_loader, and test_data_loader based on the inputs.
    Input is by default based on the configuration file, but can be manually changed in function input.
    
    Args:
    - train_dataset_labels: str representing the filepath of the train dataset labels
    - train_dataset_root_dir: str representing the filepath of the train dataset root directory
    - train_nrows: int representing the number of rows to load for the train dataset
    - test_dataset_labels: str representing the filepath of the test dataset labels
    - test_dataset_root_dir: str representing the filepath of the test dataset root directory
    - test_nrows: int representing the number of rows to load for the test dataset
    - image_file_type: str representing the image file type, e.g. ".jpg"
    - preprocess_transform: a callable function that applies preprocessing to the dataset
    - train_split_percentage: float representing the percentage of the train dataset to use
    - val_split_percentage: float representing the percentage of the validation dataset to use
    - test_split_percentage: float representing the percentage of the test dataset to use
    - batch_size: int representing the batch size
    - shuffle_train_dataloader: bool representing whether to shuffle the train dataloader
    - shuffle_val_dataloader: bool representing whether to shuffle the validation dataloader
    - num_workers: int representing the number of workers to use
    - pin_memory_train_dataloader: bool representing whether to pin memory for the train dataloader
    - is_2018_dataset: bool representing whether the dataset is from 2018
    
    Returns:
    - Tuple of train_data_loader, val_data_loader, and test_data_loader
    """
    
    assert isinstance(is_2018_dataset, bool), "Need to define dataset as either 2018 or 2019 (True or False)"



    # Load the datasets
    print("Loading datasets...")
    train_dataset_full = ISICDataset(
        csv_file=train_dataset_labels, 
        root_dir=train_dataset_root_dir, 
        transform=preprocess_transform,
        image_file_type=image_file_type,
        nrows=train_nrows
    )

    # the 2018 and 2019 dataset has some configuration differences when it comes to train, validation and test datasets
    if is_2018_dataset:
        # Splits the dataset into train and validation
        train_dataset, val_dataset = random_split(train_dataset_full, [train_split_percentage, val_split_percentage])    

        test_dataset_full = ISICDataset(
            csv_file=test_dataset_labels, 
            root_dir=test_dataset_root_dir, 
            transform=preprocess_transform,
            image_file_type=image_file_type,
            nrows=test_nrows
        )
        # added for consistency
        test_dataset = Subset(test_dataset_full, indices=[x for x in range(len(test_dataset_full))])
    else:
        train_dataset, val_dataset, test_dataset = random_split(train_dataset_full, [train_split_percentage, val_split_percentage, test_split_percentage])


    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    # Define train data loader
    print("Defining train data loader...")
    train_data_loader = data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train_dataloader,
        num_workers=num_workers,
        pin_memory=pin_memory_train_dataloader
    )

    # Define validation data loader
    print("Defining validation data loader...")
    val_data_loader = data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_val_dataloader,
        num_workers=num_workers,
        pin_memory=pin_memory_train_dataloader
    )

    # Define test data loader
    print("Defining test data loader...")
    test_data_loader = data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        pin_memory=True
    )

    # Print distribution of skin lesion categories
    train_count_dict = get_category_counts(train_data_loader)
    val_count_dict = get_category_counts(val_data_loader)
    test_count_dict = get_category_counts(test_data_loader)
    
    print("Train data loader - distribution of the skin lesion categories")
    print(train_count_dict)
    print("Validation data loader - distribution of the skin lesion categories")
    print(val_count_dict)
    print("Test data loader - distribution of the skin lesion categories")
    print(test_count_dict)
    
    return train_data_loader, val_data_loader, test_data_loader





#######################################################################
# ======================= PRIVATE FUNCTION ========================== #
#######################################################################
def _validate_model_during_training(model: torch.nn.Module, data_loader: DataLoader[Subset[ISICDataset]]) -> tuple:
    """Tests the accuracy of a trained neural network model.

    Args:
        model: The neural network model to test.
        data_loader: The data loader used for iterating over the dataset.

    Returns:
        A tuple containing the overall accuracy and F1 score of the model, and the accuracy of each of the categories of the model.
    """
    model.eval()

    # Define the list of target labels and predicted labels
    target_labels = []
    predicted_labels = []

    df: pd.DataFrame = _get_annotations_subset(data_loader) 

    # Initialize the dictionary of accuracy for each skin lesion type
    accuracy_by_type = {col: {"correct": 0, "total": 0} for col in df.columns[1:]}

    # Loop over the data in the data loader
    for i, data in tqdm(enumerate(data_loader, 0)):
        # Get the inputs and labels from the data
        inputs, labels = data

        # Move inputs and labels to the specified device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass through the model to get the predicted outputs
        outputs = model(inputs)

        # Convert the labels to a list of labels
        labels = torch.argmax(labels, 1)
        # Convert the predicted outputs to a list of labels
        predicted = torch.argmax(outputs.data, 1)

        np_labels = labels.cpu().numpy()
        np_predicted = predicted.cpu().numpy()

        # Append the target and predicted labels to their respective lists
        target_labels.extend(np_labels)
        predicted_labels.extend(np_predicted)

        # Calculate the accuracy for each skin lesion type
        for j in range(len(np_labels)):
            for k, col in enumerate(df.columns[1:]):
                if np_labels[j] == k:
                    accuracy_by_type[col]["total"] += 1
                    if np_predicted[j] == k:
                        accuracy_by_type[col]["correct"] += 1

    # Calculate the overall accuracy and F1 score
    overall_accuracy = accuracy_score(target_labels, predicted_labels)
    overall_f1_score = f1_score(target_labels, predicted_labels, average="weighted")

    # Initialize the dictionary to store the accuracy of each category
    accuracy_by_type_dict = {}

    # Calculate the accuracy for each skin lesion type
    for col in df.columns[1:]:
        if accuracy_by_type[col]["total"]:
            accuracy = accuracy_by_type[col]["correct"] / accuracy_by_type[col]["total"]
        else:
            accuracy = 0
        accuracy_by_type_dict[col] = accuracy
    
    model.train()

    return overall_accuracy, overall_f1_score, accuracy_by_type_dict



def _print_test_results(
    overall_accuracy: float,
    overall_f1_score: float,
    accuracy_by_type_dict: Dict[str, float]) -> None:
    """Prints the results of testing a trained neural network model.

    Args:
        overall_accuracy: The overall accuracy of the model.
        overall_f1_score: The overall F1 score of the model.
        accuracy_by_type_dict: A dictionary containing the accuracy for each category of the model.
    """
    
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1 score: {overall_f1_score:.4f}")
    print("Accuracy by type:")
    for category, acc in accuracy_by_type_dict.items():
        print(f" {category}: {acc:.4f}")


def _get_annotations_subset(data_loader: DataLoader[Subset[ISICDataset]]) -> pd.DataFrame:
    """Extracts the annotations subset corresponding to the indices in the given data loader.

    Args:
        data_loader: The data loader for the dataset.

    Returns:
        A pandas DataFrame containing the annotations subset, which corresponds to the respective indices.
    """
    subset: Subset = data_loader.dataset
    df: pd.DataFrame = subset.dataset.annotations
    df_subset = df.iloc[subset.indices]
    return df_subset

from typing import Dict, List, Sequence, Union, Tuple, TypeVar
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from customDataset import ISICDataset
from sklearn.metrics import accuracy_score, f1_score
from config import (
    AUGMENTED_DATASET_2019_LABELS,
    AUGMENTED_DATASET_2019_ROOT_DIR,
    AUGMENTED_TRAIN_2018_LABELS,
    AUGMENTED_TRAIN_2018_ROOT_DIR,
    DATASET_2019_LABELS,
    DATASET_2019_ROOT_DIR,
    NUM_WORKERS,
    PIN_MEMORY_TRAIN_DATALOADER,
    SHUFFLE_TRAIN_DATALOADER,
    TEST_2018_LABELS,
    TEST_2018_ROOT_DIR,
    TEST_SPLIT_2019,
    TRAIN_2018_LABELS,
    TRAIN_2018_ROOT_DIR,
    TRAIN_NROWS,
    BATCH_SIZE,
    TEST_NROWS,
    SHUFFLE_VAL_DATALOADER,
    IMAGE_FILE_TYPE,
    TRAIN_SPLIT_2018,
    TRAIN_SPLIT_2019,
    VAL_SPLIT_2018,
    VAL_SPLIT_2019
)
from torch import randperm
from torch._utils import _accumulate
import warnings
import math


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

def train_model(
        model: Module,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        criterion: Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        root_dir: str,
        writer,
        model_name: str = "",
        epoch_count: int = 20,
    ) -> Module:
    """Trains a neural network model.

    Args:
        model (Module): The neural network model to train.
        train_data_loader (DataLoader): The data loader used for iterating over the
            training dataset.
        val_data_loader (DataLoader): The data loader used for validating the training.
        criterion (Module): The loss function used for calculating the loss between
            model output and labels.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model
            parameters.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler to adjust the
            learning rate during training.
        root_dir (str): The root directory where the model and parameters will be saved.
        writer: A TensorBoard writer object for logging training metrics.
        model_name (str, optional): The type of the model being used (if applicable).
            Defaults to "".
        epoch_count (int, optional): The number of epochs to train the model. Defaults
            to 20.

    Returns:
        Module: The trained neural network model.
    """

    
    # Replace final layer with new layer that matches the number of classes in dataset
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
        for i, train_data in tqdm(enumerate(train_data_loader, 0)):
            # Get the inputs and labels from the train_data
            inputs, labels = train_data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Convert labels to a tensor of type float
            labels = labels.to(dtype=torch.float)
            labels.requires_grad = False

            outputs = model(inputs)
            
            # Calculate the loss between the model output and the labels
            loss = criterion(outputs, labels)
            
            # Backpropagate the loss through the model to calculate gradients
            loss.backward()
            
            # Update the model parameters using the gradients and the optimizer
            optimizer.step()
            
            # Add the loss for this batch to the running loss for this epoch
            running_loss += loss.item()
            optimizer.zero_grad()
        
        
        overall_accuracy, overall_f1_score = validate_model_accuracy_f1(
            model,
            val_data_loader,
            device
        )

        writer.add_scalar("Loss/train", running_loss, epoch)
        writer.add_scalar("Accuracy/validation", overall_accuracy, epoch)
        writer.add_scalar("F1 score", overall_f1_score, epoch)

        # zero gradients after each epoch
        scheduler.step()

        if epoch and epoch%10==0:
            # check the accuracy of the model
            _print_test_results(
                *validate_model_during_training(model, val_data_loader)
            )
    
    return model

def test_model(
        model: torch.nn.Module,
        data_loader: DataLoader[Subset[ISICDataset]],
        model_name: str=""
        ) -> None:
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
    for i, test_data in tqdm(enumerate(data_loader, 0)):
        # Get the inputs and labels from the test_data
        inputs, labels = test_data

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
        # Count the number of occurrences of '1' in the column
        # and add it to the dictionary
        count_dict[col] = df[col].sum()

    return count_dict
    

def random_split(
        dataset: Dataset[T],
        lengths: Sequence[Union[int, float]]
    ) -> List[Subset[T]]:
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
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = randperm(sum(lengths)).tolist()  # type: ignore[call-overload]
    return [
        Subset(dataset, indices[offset - length : offset]) 
            for offset, length in zip(_accumulate(lengths), lengths)
        ]


def get_data_loaders_2018(
        transform,
        is_augmented_dataset: bool,
        remove_print: bool=False,
        split_dataset: bool=True
    ) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader, str]:
    
    _validate_split_percentages_2018()

    labels_train = TRAIN_2018_LABELS
    root_dir_train = TRAIN_2018_ROOT_DIR
    if is_augmented_dataset:
        labels_train = AUGMENTED_TRAIN_2018_LABELS
        root_dir_train = AUGMENTED_TRAIN_2018_ROOT_DIR
    
    if split_dataset:
        data_loaders = get_data_loaders(
            *_generate_and_split_dataset_2018(
                labels_train,
                root_dir_train,
                TEST_2018_LABELS,
                TEST_2018_ROOT_DIR,
                transform
            ),
            remove_print=remove_print
        )
    
        return (*data_loaders, root_dir_train)
    else:
        data_loader = get_data_loader(
            _generate_dataset_2018(
                labels_train,
                root_dir_train,
                TEST_2018_LABELS,
                TEST_2018_ROOT_DIR,
                transform
            )
        )
        return (data_loader, root_dir_train)


def get_data_loaders_2019(
        transform,
        is_augmented_dataset: bool,
        remove_print: bool=True,
        split_dataset: bool=True
    ) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader, str]:
    
    _validate_split_percentages_2019()

    labels = DATASET_2019_LABELS
    root_dir = DATASET_2019_ROOT_DIR
    if is_augmented_dataset:
        labels = AUGMENTED_DATASET_2019_LABELS
        root_dir = AUGMENTED_DATASET_2019_ROOT_DIR

    if split_dataset:
        data_loaders = get_data_loaders(
            *_generate_and_split_dataset_2019(
                labels,
                root_dir,
                transform
            ),
            remove_print=remove_print
        )

        return (*data_loaders, root_dir)
    else:
        data_loader = get_data_loader(
            _generate_dataset_2019(
                labels,
                root_dir,
                TEST_2018_LABELS,
                TEST_2018_ROOT_DIR,
                transform
            )
        )
        return (data_loader, root_dir)


def get_data_loader(
        dataset: data.Dataset, 
        remove_print: bool=False,
) -> data.DataLoader:
    """
    Returns a PyTorch DataLoader for a given dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to create a dataloader for.
    
    Returns:
        torch.utils.data.DataLoader: A dataloader for the given dataset.
    """
    
    dataloader = data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE_TRAIN_DATALOADER,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY_TRAIN_DATALOADER
    )
    if not remove_print:
        # Print distribution of skin lesion categories
        count_dict = get_category_counts(dataloader)
        
        print("Dataloader - distribution of the skin lesion categories")
        print(count_dict)

    return dataloader


def get_data_loaders(
    train_dataset: data.Dataset,
    val_dataset: data.Dataset,
    test_dataset: data.Dataset,
    remove_print: bool=False
) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """
    Returns train_data_loader, val_data_loader, and test_data_loader.

    Args:
        train_dataset: A PyTorch dataset containing the training data.
        val_dataset: A PyTorch dataset containing the validation data.
        test_dataset: A PyTorch dataset containing the test data.

    Returns:
        A tuple containing three PyTorch data loaders:
            train_data_loader, val_data_loader, and test_data_loader.
    """
    

    # Define train data loader
    if not remove_print: print("Defining train data loader...")
    train_data_loader = data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE_TRAIN_DATALOADER,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY_TRAIN_DATALOADER
    )

    # Define validation data loader
    if not remove_print: print("Defining validation data loader...")
    val_data_loader = data.DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=SHUFFLE_VAL_DATALOADER,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY_TRAIN_DATALOADER
    )

    # Define test data loader
    if not remove_print: print("Defining test data loader...")
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
    if not remove_print:
        print(f"Train dataset length: {len(train_dataset)}")
        print(f"Validation dataset length: {len(val_dataset)}")
        print(f"Test dataset length: {len(test_dataset)}")
        print("Train data loader - distribution of the skin lesion categories")
        print(train_count_dict)
        print("Validation data loader - distribution of the skin lesion categories")
        print(val_count_dict)
        print("Test data loader - distribution of the skin lesion categories")
        print(test_count_dict)
    
    return train_data_loader, val_data_loader, test_data_loader

def get_data_loaders_by_year(
        year,
        transform,
        is_augmented_dataset,
        remove_print: bool=False,
        split_dataset: bool=True
) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader, str] | tuple[data.DataLoader, str]:
    if year == "2018":
        return get_data_loaders_2018(
            transform=transform,
            is_augmented_dataset=is_augmented_dataset,
            remove_print=remove_print,
            split_dataset=split_dataset
        )
    elif year == "2019":
        return get_data_loaders_2019(
            transform=transform,
            is_augmented_dataset=is_augmented_dataset,
            remove_print=remove_print,
            split_dataset=split_dataset
        )
    else:
        raise Exception("Need to choose dataset year...")



#######################################################################
# ======================= PRIVATE FUNCTION ========================== #
#######################################################################
def validate_model_during_training(
        model: torch.nn.Module,
        data_loader: DataLoader[Subset[ISICDataset]]
) -> tuple:
    """Tests the accuracy of a trained neural network model.

    Args:
        model: The neural network model to test.
        data_loader: The data loader used for iterating over the dataset.

    Returns:
        A tuple containing the overall accuracy and F1 score of the model, and the
            accuracy of each of the categories of the model.
    """
    model.eval()

    # Define the list of target labels and predicted labels
    target_labels = []
    predicted_labels = []

    df: pd.DataFrame = _get_annotations_subset(data_loader) 

    # Initialize the dictionary of accuracy for each skin lesion type
    accuracy_by_type = {col: {"correct": 0, "total": 0} for col in df.columns[1:]}

     # Disable gradient calculation to save memory and speed up validation
    with torch.no_grad():
        for i, (inputs, labels) in tqdm(enumerate(data_loader, 0)):

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
            accuracy_tot = accuracy_by_type[col]["total"]
            if accuracy_by_type[col]["total"]:
                accuracy = accuracy_by_type[col]["correct"]/accuracy_tot
                accuracy = 0
            accuracy_by_type_dict[col] = accuracy
        
        model.train()

    return overall_accuracy, overall_f1_score, accuracy_by_type_dict

def validate_model_accuracy_f1(model, val_data_loader, device):
    model.eval()  # Set the model to evaluation mode

    true_labels = []
    predicted_labels = []
    # Disable gradient calculation to save memory and speed up validation
    with torch.no_grad(): 
        for val_data in val_data_loader:
            inputs, labels = val_data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get class with the highest probability
            _, labels = torch.max(labels, 1)  # Get class with the highest probability

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    # Calculate the overall accuracy and F1 score
    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    overall_f1_score = f1_score(true_labels, predicted_labels, average='weighted')

    model.train()  # Set the model back to training mode

    return overall_accuracy, overall_f1_score


def _print_test_results(
    overall_accuracy: float,
    overall_f1_score: float,
    accuracy_by_type_dict: Dict[str, float]) -> None:
    """
    Prints the results of testing a trained neural network model.

    Args:
        overall_accuracy: The overall accuracy of the model.
        overall_f1_score: The overall F1 score of the model.
        accuracy_by_type_dict: The accuracy for each category of the model.
    """
    
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    print(f"Overall F1 score: {overall_f1_score:.4f}")
    print("Accuracy by type:")
    for category, acc in accuracy_by_type_dict.items():
        print(f" {category}: {acc:.4f}")


def _get_annotations_subset(
        data_loader: DataLoader[Subset[ISICDataset]]
    ) -> pd.DataFrame:
    """
    Extracts the annotations subset to the indices in the given dataloader.

    Args:
        data_loader: The data loader for the dataset.

    Returns:
        A pandas DataFrame containing the annotations subset, which corresponds to the
            respective indices.
    """
    subset: Subset = data_loader.dataset
    df: pd.DataFrame = subset.dataset.annotations
    df_subset = df.iloc[subset.indices]
    return df_subset

def _validate_split_percentages_2019() -> None:
    """
    Validates that the sum of train, validation, and test split percentages equals 1.0.

    Raises:
        AssertionError: If the sum of the percentages is not equal to 1.0.

    Returns:
        None.
    """
    assertion_message: str = (
        "The total of train, validation and test percentage should be equal to 1.0"
    )
    split_percentage_sum = sum([
        TRAIN_SPLIT_2019,
        VAL_SPLIT_2019,
        TEST_SPLIT_2019
    ])
    assert split_percentage_sum == 1.0, assertion_message

def _validate_split_percentages_2018() -> None:
    """
    Validates that the sum of train and validation split percentages equals 1.0.
    the 2018 dataset already has a test dataset generated for them.

    Raises:
        AssertionError: If the sum of the percentages is not equal to 1.0.

    Returns:
        None.
    """
    assertion_message: str = (
        "The total of train and validation percentage should be equal to 1.0"
    )
    split_percentage_sum = sum([
        TRAIN_SPLIT_2018,
        VAL_SPLIT_2018
    ])
    assert split_percentage_sum == 1.0, assertion_message


def _generate_and_split_dataset_2018(
        labels_train: str,
        root_dir_train: str,
        labels_test: str,
        root_dir_test: str,
        transform
    ) -> tuple[Subset, Subset, Subset]:
    """
    Splits the dataset into train, validation and test based on the provided split
        percentages.
    """

    _validate_split_percentages_2018()

    print("Loading datasets...")
    train_dataset_full = ISICDataset(
        csv_file=labels_train, 
        root_dir=root_dir_train, 
        transform=transform,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TRAIN_NROWS
    )

    test_dataset_full = ISICDataset(
        csv_file=labels_test, 
        root_dir=root_dir_test, 
        transform=transform,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TEST_NROWS
    )

    # splits the dataset
    train_dataset, val_dataset = random_split(
        train_dataset_full, [TRAIN_SPLIT_2018, VAL_SPLIT_2018]
    )
    test_dataset = Subset(
        test_dataset_full, indices=[x for x in range(len(test_dataset_full))]
    )
    
    return train_dataset, val_dataset, test_dataset

def _generate_and_split_dataset_2019(
        labels: str,
        root_dir: str,
        transform
    ) -> tuple[Subset, Subset, Subset]:

    _validate_split_percentages_2019()
    
    print("Loading datasets...")
    train_dataset_full = ISICDataset(
        csv_file=labels, 
        root_dir=root_dir, 
        transform=transform,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TRAIN_NROWS
    )

    # splits the dataset
    train_validation_test_dataset: tuple = random_split(
        train_dataset_full, [
            TRAIN_SPLIT_2019, VAL_SPLIT_2019, TEST_SPLIT_2019
        ]
    )
    return train_validation_test_dataset



def _generate_dataset_2018(
        labels_train: str,
        root_dir_train: str,
        labels_test: str,
        root_dir_test: str,
        transform
    ) -> Subset:

    print("Loading datasets...")
    train_dataset_full = ISICDataset(
        csv_file=labels_train, 
        root_dir=root_dir_train, 
        transform=transform,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TRAIN_NROWS
    )

    train_dataset = Subset(
        train_dataset_full, indices=[x for x in range(len(train_dataset_full))]
    )
    return train_dataset

def _generate_dataset_2019(
        labels: str,
        root_dir: str,
        transform
    ) -> Subset:
    
    print("Loading datasets...")
    train_dataset_full = ISICDataset(
        csv_file=labels, 
        root_dir=root_dir, 
        transform=transform,
        image_file_type=IMAGE_FILE_TYPE,
        nrows=TRAIN_NROWS
    )

    train_dataset = Subset(
        train_dataset_full, indices=[x for x in range(len(train_dataset_full))]
    )

    return train_dataset
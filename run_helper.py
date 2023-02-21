import sys
from typing import Dict
import torch
import numpy as np
from torch.utils.data import Subset
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from customDataset import ISICDataset # Import the Pytorch library
from sklearn.metrics import accuracy_score, f1_score

def train_model(
        model: Module,
        train_dataset_subset: Subset[ISICDataset],
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        criterion: Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        model_name: str = "",
        epoch_count: int = 20,
        requires_grad: bool = True) -> Module:
    """Trains a neural network model by retraining a model with already existing weights form Image Net.

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
    train_dataset = train_dataset_subset.dataset
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
            labels = torch.tensor(labels, dtype=torch.float)

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

    
        # check the accuracy of the model
        overall_accuracy, overall_f1_score, accuracy_by_type_dict = _validate_model_during_training(model, val_data_loader)
        _print_test_results(overall_accuracy, overall_f1_score, accuracy_by_type_dict)

        # Print the average loss for this epoch
        print('Epoch {} loss: {:.4f}'.format(epoch + 1, running_loss / (i + 1)))

    # Print a message indicating that training has finished
    print('Finished training')


    return model

def test_model(model, dataset, data_loader, model_name: str=""):
    """Tests a neural network model on a dataset and prints the accuracy and F1 score.

    Args:
        model: The neural network model to test.
        dataset: The dataset used for testing.
        data_loader: The data loader used for iterating over the dataset.
    """
    # Put the model in evaluation mode to turn off dropout and batch normalization
    model.eval()

    # Define the list of target labels and predicted labels
    target_labels = []
    predicted_labels = []

    # Initialize the dictionary of accuracy for each skin lesion type
    accuracy_by_type = {col: {"correct": 0, "total": 0} for col in dataset.annotations.columns[1:]}

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
            for k, col in enumerate(dataset.annotations.columns[1:]):
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
    for col in dataset.annotations.columns[1:]:
        if accuracy_by_type[col]["total"]:
            accuracy = accuracy_by_type[col]["correct"] / accuracy_by_type[col]["total"]
        else:
            accuracy = 0
        print("{} accuracy: {:.4f}".format(col, accuracy))







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

    # Initialize the dictionary of accuracy for each skin lesion type
    accuracy_by_type = {col: {"correct": 0, "total": 0} for col in data_loader.dataset.dataset.annotations.columns[1:]}

    # Loop over the data in the data loader
    for i, data in enumerate(data_loader, 0):
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
            for k, col in enumerate(data_loader.dataset.dataset.annotations.columns[1:]):
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
    for col in data_loader.dataset.dataset.annotations.columns[1:]:
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
    for category, acc_dict in accuracy_by_type_dict.items():
        acc = acc_dict["correct"] / acc_dict["total"] if acc_dict["total"] != 0 else 0
        print(f" {category}: {acc:.4f}")

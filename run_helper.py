import sys
from typing import Dict
import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
from customDataset import ISICDataset # Import the Pytorch library
from sklearn.metrics import accuracy_score, f1_score


def train_model_finetuning(
        model: Module,
        dataset: ISICDataset,
        data_loader: DataLoader,
        criterion: Module, 
        optimizer: torch.optim.Optimizer,
        model_name: str = "",
        epoch_count: int = 20) -> Module:
    """Trains a neural network model by fine-tuning the last layer.

    Args:
        model: The neural network model to train.
        dataset: The dataset used for training.
        data_loader: The data loader used for iterating over the dataset.
        criterion: The loss function used for calculating the loss between model output and labels.
        optimizer: The optimizer used for updating the model parameters.
        model_name: The type of the model being used (if applicable).
        epoch_count: The number of epochs to train the model.

    Returns:
        The trained neural network model.
    """
    # Freeze the model parameters to prevent backpropagation
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer with a new layer that matches the number of classes in the dataset
    num_classes = len(dataset.annotations.columns)-1
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the GPU if GPU is available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Is CUDA available:", torch.cuda.is_available())
    model.to(device)


    # Loop over the number of epochs
    for epoch in range(epoch_count):
        # Initialize the running loss for this epoch
        running_loss = 0.0
        
        # Loop over the data in the data loader
        for i, data in enumerate(data_loader, 0):
            # Get the inputs and labels from the data
            inputs, labels = data
            
            # Convert labels to a tensor of type float
            labels = torch.tensor(labels, dtype=torch.float)
            
            # Clear the gradients in the optimizer
            optimizer.zero_grad()
            
            # Move inputs and labels to the specified device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
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

        # Print the average loss for this epoch
        print('Epoch {} loss: {:.4f}'.format(epoch + 1, running_loss / (i + 1)))

    # Print a message indicating that training has finished
    print('Finished training')


    return model

def test_model(model, dataset, data_loader):
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
    for i, data in enumerate(data_loader, 0):
        # Get the inputs and labels from the data
        inputs, labels = data

        # Move inputs and labels to the specified device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Pass inputs through the model to get the predicted outputs
        outputs = model(inputs)

        # Convert the predicted outputs to a list of labels
        _, predicted = torch.max(outputs.data, 1)

        # Append the target and predicted labels to their respective lists
        target_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

        # Calculate the accuracy for each skin lesion type
        for j in range(len(labels)):
            for k, col in enumerate(dataset.annotations.columns[1:]):
                if labels[j][k] == 1:
                    accuracy_by_type[col]["total"] += 1
                    if predicted[j] == k:
                        accuracy_by_type[col]["correct"] += 1

    # Calculate the overall accuracy and F1 score
    overall_accuracy = accuracy_score(target_labels, predicted_labels)
    overall_f1_score = f1_score(target_labels, predicted_labels, average="weighted")

    # Print the overall accuracy and F1 score
    print("Overall accuracy: {:.4f}".format(overall_accuracy))
    print("Overall F1 score: {:.4f}".format(overall_f1_score))

    # Print the accuracy for each skin lesion type
    for col in dataset.annotations.columns[1:]:
        accuracy = accuracy_by_type[col]["correct"] / accuracy_by_type[col]["total"]
        print("{} accuracy: {:.4f}".format(col, accuracy))

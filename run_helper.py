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


def test_model(model: Module, dataset: ISICDataset, data_loader: DataLoader) -> Dict:
    """Tests a neural network model on a separate dataset.

    Args:
        model: The trained neural network model.
        dataset: The dataset used for testing.
        data_loader: The data loader used for iterating over the dataset.

    Returns:
        A dictionary containing the accuracy, F1 score, and accuracy for different categories.
    """
    # Set the model to evaluation mode
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize variables for computing accuracy and F1 score
    true_labels = []
    predicted_labels = []

    # Initialize variables for computing accuracy for different categories
    category_correct = {}
    category_total = {}

    # Loop over the data in the data loader
    for i, data in enumerate(data_loader, 0):
        # Get the inputs and labels from the data
        inputs, labels = data

        # Convert labels to a tensor of type float
        labels = torch.tensor(labels, dtype=torch.float)

        # Move inputs and labels to the specified device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Pass inputs through the model to get the predicted labels
        outputs = model(inputs)
        predicted = torch.round(outputs)

        # Convert predicted labels and true labels to numpy arrays and append to the variables
        predicted_labels += predicted.cpu().numpy().tolist()
        true_labels += labels.cpu().numpy().tolist()

        # Compute accuracy for different categories
        for j, category in enumerate(dataset.annotations.columns[1:]):
            if category not in category_correct:
                category_correct[category] = 0
                category_total[category] = 0

            category_predicted = predicted[:, j].cpu().numpy().tolist()
            category_true = labels[:, j].cpu().numpy().tolist()
            category_accuracy = accuracy_score(category_true, category_predicted)
            category_correct[category] += np.sum(np.logical_and(category_predicted, category_true))
            category_total[category] += np.sum(category_true)

    # Compute overall accuracy and F1 score
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    # Compute accuracy for different categories
    category_accuracy = {}
    for category in dataset.annotations.columns[1:]:
        category_accuracy[category] = category_correct[category] / category_total[category]

    # Print the accuracy, F1 score, and accuracy for different categories
    print('Accuracy: {:.4f}'.format(accuracy))
    print('F1 score: {:.4f}'.format(f1))
    print('Accuracy by category:')
    for category in category_accuracy:
        print('{}: {:.4f}'.format(category, category_accuracy[category]))

    # Return a dictionary containing the accuracy, F1 score, and accuracy for different categories
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'category_accuracy': category_accuracy
    }

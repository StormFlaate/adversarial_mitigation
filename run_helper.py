import sys
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from customDataset import ISICDataset # Import the Pytorch library



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
    device = torch.device("cude" if torch.cuda.is_available() else "cpu")
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
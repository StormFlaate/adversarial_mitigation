import torch

from customDataset import ISICDataset # Import the Pytorch library



def train_model_finetuning(
        model,
        dataset: ISICDataset,
        data_loader,
        criterion, 
        optimizer, 
        epoch_count:int=20):
    

    # Freeze the model parameters to prevent backpropagation
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer with a new layer that matches the number of classes in the dataset
    num_classes = len(dataset.annotations.columns)-1
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # Move the model to the GPU if GPU is available 
    device = torch.device("cude" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for epoch in range(epoch_count):
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            labels = torch.tensor(labels, dtype=torch.float)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, x = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print('Epoch {} loss: {:.4f}'.format(epoch + 1, running_loss / (i + 1)))

    print('Finished training')

    return model
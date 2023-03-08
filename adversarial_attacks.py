import numpy as np
import torch
import torchattacks
from tqdm import tqdm
from misc_helper import get_trained_or_default_model
from train_model_helper import get_data_loaders

print("get data loaders...")
train_data_loader, val_data_loader, test_data_loader = get_data_loaders(batch_size=1, num_workers=1)
print("get trained or deafult model...")
model = get_trained_or_default_model(model_file_name="resnet18_data_ISIC2018_Training_Input_2023-03-03__ff2.pt")


attack = torchattacks.FGSM(model, eps=2/255)

for index, (input, true_label) in  tqdm(enumerate(train_data_loader)):

    # Move inputs and labels to the specified device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    true_label = true_label.to(device)

    # turns 1-dimensional list into 0-dimensional scalar
    true_label_argmax = torch.argmax(true_label, 1)

    # runs the optimzation on the input given the true label
    adversarial_input = attack(input, true_label_argmax)
    predicted_label = model(adversarial_input)


    np_true_label = true_label_argmax.detach().cpu().numpy()
    np_predicted_label = predicted_label.detach().cpu().numpy()
    

    print("true_label_argmax", true_label_argmax)
    print("predicted_label", predicted_label)
    print("np_true_label", np_true_label)
    print("np_predicted_label", np_predicted_label)

    print(f"True: {np.argmax(np_true_label)}")
    print(f"Pred: {np.argmax(np_predicted_label)}")
    break
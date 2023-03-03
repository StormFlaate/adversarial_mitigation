import torch
import torchattacks
from misc_helper import get_trained_or_default_model
from run_helper import get_data_loaders


train_data_loader, val_data_loader, test_data_loader = get_data_loaders()
model = get_trained_or_default_model(model_file_name="resnet18_augmented_data_ISIC2018_Training_Input_2023-02-22__f69.pt")

attack = torchattacks.FGSM(model, eps=2/255)

for index, (input, true_label) in  enumerate(train_data_loader):
    label_arg = torch.argmax(true_label, 1)
    adv_input = attack(input, label_arg)
    predicted_label = model(adv_input)
    print(torch.argmax(predicted_label.data), torch.argmax(true_label))
    break
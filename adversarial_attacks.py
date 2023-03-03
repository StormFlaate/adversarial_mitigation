import torch
import torchattacks
from misc_helper import get_trained_or_default_model
from run_helper import get_data_loaders


train_data_loader, val_data_loader, test_data_loader = get_data_loaders(batch_size=1)
model = get_trained_or_default_model(model_file_name="resnet18_data_ISIC2018_Training_Input_2023-03-03__ff2.pt")

attack = torchattacks.FGSM(model, eps=2/255)

for index, (input, true_label) in  enumerate(train_data_loader):
    label_arg = torch.argmax(true_label, 1)
    adv_input = attack(input, label_arg)
    predicted_label = model(adv_input)

    np_label = label_arg.detach().cpu().numpy()
    np_predicted = predicted_label.detach().cpu().numpy()
    print(f"True: {torch.argmax(np_label)}")
    print(f"Pred: {torch.argmax(np_predicted)}")
    break
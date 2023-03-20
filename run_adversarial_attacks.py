import numpy as np
import torchattacks
from tqdm import tqdm
from adversarial_attacks_helper import generate_adversarial_input
from misc_helper import get_trained_or_default_model
from train_model_helper import get_data_loaders

print("get data loaders...")
train_data_loader, val_data_loader, test_data_loader = get_data_loaders(
    batch_size=1, 
    num_workers=1)

print("get trained or deafult model...")
model = get_trained_or_default_model(
    model_file_name="resnet18_data_ISIC2018_Training_Input_2023-03-03__ff2.pt")


attack = torchattacks.FGSM(model, eps=2/255)

for index, (input, true_label) in  tqdm(enumerate(train_data_loader)):

    adversarial_input = generate_adversarial_input(input, true_label, attack)

    predicted_label = model(adversarial_input)


    np_true_label = true_label.detach().cpu().numpy()
    np_predicted_label = predicted_label.detach().cpu().numpy()
    

    print("true_label_argmax", true_label)
    print("predicted_label", predicted_label)
    print("np_true_label", np_true_label)
    print("np_predicted_label", np_predicted_label)

    print(f"True: {np.argmax(np_true_label)}")
    print(f"Pred: {np.argmax(np_predicted_label)}")
    break
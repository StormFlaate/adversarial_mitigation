import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from config import MODEL_NAME

model = torch.hub.load('pytorch/vision:v0.10.0', MODEL_NAME, pretrained=True)
model.eval()

# Load and preprocess the image
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)

feature_maps = []

def hook(module, input, output):
    feature_maps.append(output)

if isinstance(model, nn.Sequential):
    first_layer = model[0]
else:
    first_layer = model.features[0]

handle = first_layer.register_forward_hook(hook)

with torch.no_grad():
    output = model(input_tensor)

handle.remove()
print("Feature maps:", feature_maps)

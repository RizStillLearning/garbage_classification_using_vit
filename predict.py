import torch
import torch.nn.functional as F
import argparse
from PIL import Image
from model import build_model, load_model
from utils import get_device, get_transform, get_config

model = build_model(num_classes=10)
device = get_device()
model.to(device)

config = get_config()
best_model_path = config['best_model_path']
save_config = load_model(model, best_model_path, device)
classes = save_config['classes']

parser = argparse.ArgumentParser(description='Predict class for an input image')
parser.add_argument('image_path', type=str, help='Path to the input image')
args = parser.parse_args()

def predict(model, device, image_path, classes):
    model.eval()
    transform = get_transform('test')
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = classes[predicted.item()]
        probability = F.softmax(outputs, dim=1)[0][predicted.item()].item()
    
    return predicted_class, probability

if __name__ == '__main__':
    predicted_class, probability = predict(model, device, args.image_path, classes)
    print(f"Predicted class: {predicted_class}, Probability: {probability:.4f}")
import torch
import timm
import os
from utils import get_config
    
def build_model(num_classes=10):
    config = get_config()
    model_name = config['model_name']
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def save_model(model, save_config: dict, file_name='best_model.pth'):
    os.makedirs(get_config()['model_dir'], exist_ok=True)
    save_config['model_state_dict'] = model.state_dict()
    file_path = os.path.join(get_config()['model_dir'], file_name)
    torch.save(save_config, file_path)

def load_model(model, file_name, device):
    file_path = os.path.join(get_config()['model_dir'], file_name)
    save_config = torch.load(file_path, map_location=device)
    model.load_state_dict(save_config['model_state_dict'])
    model.to(device)
    return save_config
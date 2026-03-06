import torch
import timm
from utils import get_config
    
def build_model(num_classes=10):
    config = get_config()
    model_name = config['model_name']
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def save_model(model, save_config: dict, file_path='best_model.pth'):
    save_config['model_state_dict'] = model.state_dict()
    torch.save(save_config, file_path)

def load_model(model, path, device):
    save_config = torch.load(path, map_location=device)
    model.load_state_dict(save_config['model_state_dict'])
    model.to(device)
    return save_config
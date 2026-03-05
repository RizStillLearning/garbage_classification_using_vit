import torch
import torch.nn as nn
import timm
    
def build_model(model_name='vit_base_patch16_224', num_classes=10):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def save_model(model, save_config: dict, file_path='best_model.pth'):
    save_config['model_state_dict'] = model.state_dict()
    torch.save(save_config, file_path)

def load_model(model, path, device):
    save_config = torch.load(path, map_location=device)
    model.load_state_dict(save_config['model_state_dict'])
    model.to(device)
    return model
import torch
import yaml
import os
import csv
import json
import pandas as pd
from pathlib import Path
from torchvision import transforms
from typing import Literal

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config(file_path='config.yaml'):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_transform(mode: Literal['train', 'val', 'test']):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_size = get_config()['image_size']

    transforms_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),  # Random crop and resize
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),  # Random rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color augmentation
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Slight blur
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Random erasing
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    return transforms_dict[mode]

def get_target_transform():
    return transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, val_loss, file_name='checkpoint.pth'):
    os.makedirs(get_config()['model_dir'], exist_ok=True)
    file_path = os.path.join(get_config()['model_dir'], file_name)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }
    torch.save(checkpoint, file_path)

def load_checkpoint(model, optimizer, file_name):
    file_path = os.path.join(get_config()['model_dir'], file_name)
    checkpoint = torch.load(file_path, weights_only=False, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['val_loss']

def write_training_log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, file_name='training_log.csv'):
    os.makedirs(get_config()['output_dir'], exist_ok=True)
    file_path = os.path.join(get_config()['output_dir'], file_name)
    file_open_mode = 'a' if epoch > 1 else 'w'
    with open(file_path, file_open_mode, newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        if epoch == 1 or not Path(file_path).exists():
            log_writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'Learning Rate'])
        log_writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])

def save_classification_report(report, file_name):
    os.makedirs(get_config()['output_dir'], exist_ok=True)
    file_path = os.path.join(get_config()['output_dir'], file_name)
    with open(file_path, 'w') as f:
        json.dump(report, f, indent=4)

def save_confusion_matrix(conf_matrix, file_name):
    file_path = os.path.join(get_config()['output_dir'], file_name)
    df = pd.DataFrame(conf_matrix)
    df.to_csv(file_path, index=True)
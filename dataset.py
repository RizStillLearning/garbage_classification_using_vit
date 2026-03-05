import kagglehub
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utils import get_config, get_transform, get_target_transform
from typing import Literal

# Custom Dataset class for loading images and labels
class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.images = dataset['images']
        self.labels = dataset['labels']
        self.transform = transform # Transformations for images (e.g., resizing, normalization)
        self.target_transform = target_transform # Transformations for labels (e.g., encoding)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

def load_dataset_from_kaggle(dataset_name, dataset_dir):
    # Load the dataset using kagglehub
    path = kagglehub.dataset_download(dataset_name)
    # Read the images and labels from the dataset
    images = []
    labels = []

    dataset_directory = Path(f"{path}/{dataset_dir}")
    for dir in dataset_directory.rglob('*'):
        if dir.is_file():
            img = Image.open(dir).convert('RGB')
            label = dir.parent.name
            images.append(img)
            labels.append(label)

    # Create a mapping from class labels to indices
    classes = list(sorted(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    labels = [label_to_index[label] for label in labels]
    df = pd.DataFrame({'image': images, 'label': labels})
    return df, classes

def split_dataset(dataframe, test_size=0.2, random_state=42):
    # Split the dataset into training, validation, and test sets
    train_df, temp_df = train_test_split(dataframe, test_size=test_size, random_state=random_state, stratify=dataframe['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df['label'])

    # Extract images and labels for each set
    train_images = train_df['image'].tolist()
    train_labels = train_df['label'].tolist()
    val_images = val_df['image'].tolist()
    val_labels = val_df['label'].tolist()
    test_images = test_df['image'].tolist()
    test_labels = test_df['label'].tolist()
    # Create dictionaries for each dataset
    train_dataset = {
        'images': train_images,
        'labels': train_labels
    }

    val_dataset = {
        'images': val_images,
        'labels': val_labels
    }

    test_dataset = {
        'images': test_images,
        'labels': test_labels
    }

    return train_dataset, val_dataset, test_dataset

def build_dataloaders(dataset, mode: Literal['train', 'val', 'test'], target_transform):
    # Create DataLoader instances based on the passed datasets and transformations
    config = get_config()
    batch_size = config['batch_size']
    transform = get_transform(mode)

    image_dataset = ImageDataset(dataset, transform=transform, target_transform=target_transform)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=4, pin_memory=True)
    return dataloader
import torch
import torch.nn as nn
import os
from model import build_model, load_model
from utils import get_config, get_device
from huggingface_hub import PyTorchModelHubMixin

class DeployModel(nn.Module, PyTorchModelHubMixin, repo_url="https://huggingface.co/harriskr14/garbage_classifier", pipeline_tag="text-classification", license="mit"):
    def __init__(self, num_classes=10):
        super(DeployModel, self).__init__()
        self.model = build_model(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    config = get_config()
    num_classes = config['num_classes']
    model = DeployModel(num_classes=num_classes)
    # Load the best model weights
    device = get_device()
    save_config = load_model(model.model, file_name='best_model.pth', device=device)
    # Save the model to Hugging Face Hub
    model.push_to_hub("harriskr14/garbage_classifier")
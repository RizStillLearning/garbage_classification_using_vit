import torch
import torch.optim as optim
import torch.nn as nn
from dataset import load_dataset_from_kaggle, split_dataset, build_dataloaders
from utils import get_device, get_transform, get_target_transform, seed_everything, get_config, save_checkpoint, load_checkpoint
from model import build_model
from visualization import visualize_dataset
from train import train_epoch, validate, evaluate_model
from pathlib import Path

# Additional imports for learning rate scheduler
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def main():
    torch.cuda.empty_cache()

    seed = 42
    seed_everything(seed)

    print("Loading dataset...")
    df, classes = load_dataset_from_kaggle(dataset_name="sumn2u/garbage-classification-v2", dataset_dir="original")
    train_dataset, val_dataset, test_dataset = split_dataset(df)

    print("Visualizing dataset...")
    visualize_dataset(train_dataset, classes, file_name='train_label_distribution.png')
    visualize_dataset(val_dataset, classes, file_name='val_label_distribution.png')
    visualize_dataset(test_dataset, classes, file_name='test_label_distribution.png')

    print("Building dataloaders...")
    transform = get_transform()
    target_transform = get_target_transform()
    train_loader, val_loader, test_loader = build_dataloaders(train_dataset, val_dataset, test_dataset, transform, target_transform)

    config = get_config()

    print("Training model...")
    model = build_model(num_classes=len(classes))
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    best_val_loss = float('inf')
    cur_epoch = 1

    best_model_path = 'checkpoint.pth'

    current_dir = Path.cwd()
    checkpoint_file = list(current_dir.glob(best_model_path))
    # Check if checkpoint file exists and load it
    if checkpoint_file:
        cur_epoch, best_val_loss, _ = load_checkpoint(model, optimizer, best_model_path)
        print("Loaded model from checkpoint")
        print(f"Current epoch: {cur_epoch}, Current best validation loss: {best_val_loss:.4f}")

    # Create learning rate scheduler with warmup
    num_epochs = config['num_epochs']
    num_warmup_steps = len(train_loader)
    num_training_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    for params in model.parameters():
        params.requires_grad = True

    print(f"Starting training for {num_epochs} epochs on {device}...")
    print("-" * 80)

    for epoch in range(cur_epoch, num_epochs+1):
        train_loss, train_acc = train_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)

        # Update learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        log = f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, LR: {current_lr:.6f}"
        print(log)

        mode = "w"
        if epoch > 1:
            mode = "a"
        
        with open("outputs/training_log.txt", mode) as f:
            f.write(log + "\n")

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, file_path=best_model_path)
            print(f"  ✓ Best model saved! (val_loss: {val_loss:.4f})")

    print("-" * 80)
    print("Training completed!")

    # Evaluate on test set with best model
    print("\nLoading best model and evaluating on test set...")
    load_checkpoint(model, optimizer, best_model_path)
    test_loss, test_acc = evaluate_model(model, device, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
import torch
import torch.optim as optim
import torch.nn as nn
import gc
import csv
from dataset import load_dataset_from_kaggle, split_dataset, build_dataloaders
from utils import get_device, get_transform, get_target_transform, seed_everything, get_config, save_checkpoint, load_checkpoint
from model import build_model, save_model, load_model
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
    seed = 42
    seed_everything(seed)

    print("Loading dataset...")
    df, classes = load_dataset_from_kaggle(dataset_name="sumn2u/garbage-classification-v2", dataset_dir="original")
    train_dataset, val_dataset, test_dataset = split_dataset(df)

    print("Building dataloaders...")
    target_transform = get_target_transform()
    train_dataloader = build_dataloaders(train_dataset, mode='train', target_transform=target_transform)
    val_dataloader = build_dataloaders(val_dataset, mode='val', target_transform=target_transform)
    test_dataloader = build_dataloaders(test_dataset, mode='test', target_transform=target_transform)

    config = get_config()

    print("Training model...")
    model = build_model(num_classes=len(classes))
    device = get_device()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    best_val_loss = float('inf')
    best_model = build_model(num_classes=len(classes))
    cur_epoch = 1

    best_model_path = config['checkpoint_path']
    output_log_dir = config['output_log_dir']

    # Check if checkpoint file exists and load it
    if Path(best_model_path).exists():
        cur_epoch, best_val_loss = load_checkpoint(model, optimizer, best_model_path)
        cur_epoch += 1  # Start from the next epoch
        best_model.load_state_dict(model.state_dict())
        print("Loaded model from checkpoint")
        print(f"Current epoch: {cur_epoch}, Current best validation loss: {best_val_loss:.4f}")

    # Create learning rate scheduler with warmup
    num_epochs = config['num_epochs']
    num_warmup_steps = len(train_dataloader)
    num_training_steps = len(train_dataloader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    for params in model.parameters():
        params.requires_grad = True

    print(f"Starting training for {num_epochs} epochs on {device}...")
    print("-" * 80)

    for epoch in range(cur_epoch, num_epochs+1):
        train_loss, train_acc = train_epoch(model, device, train_dataloader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_dataloader, criterion)

        # Update learning rate scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        log = f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, LR: {current_lr:.6f}"
        print(log)
        # Save training log to CSV
        file_open_mode = 'a' if epoch > 1 else 'w'
        with open(output_log_dir, file_open_mode, newline='') as csvfile:
            log_writer = csv.writer(csvfile)
            if epoch == 1 or not Path(output_log_dir).exists():
                log_writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'Learning Rate'])
            log_writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}", f"{current_lr:.6f}"])
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model.load_state_dict(model.state_dict())
            print(f"  ✓ Best model saved! (val_loss: {val_loss:.4f})")
            
        save_checkpoint(best_model, optimizer, epoch, best_val_loss, best_model_path)
        print("  ✓ Checkpoint saved!")

        gc.collect()
        torch.cuda.empty_cache()

    print("-" * 80)
    print("Training completed!")

    # Evaluate on test set with best model
    print("\nLoading best model and evaluating on test set...")
    load_model(model, best_model_path, device)
    test_loss, test_acc = evaluate_model(model, device, test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the best model for future inference
    save_config = {
        'classes': classes,
        'image_size': config['image_size']
    }
    save_model(best_model, save_config, file_path='best_model.pth')
    print("Best model saved to 'best_model.pth'")

if __name__ == '__main__':
    main()
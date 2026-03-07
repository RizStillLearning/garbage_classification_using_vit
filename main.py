import torch
import torch.optim as optim
import torch.nn as nn
import gc
import os
from dataset import load_dataset_from_kaggle, split_dataset, build_dataloaders
from utils import get_device, get_target_transform, seed_everything, get_config, save_checkpoint, load_checkpoint, write_training_log, save_classification_report, save_confusion_matrix
from model import build_model, save_model, load_model
from train import train_epoch, validate, evaluate_model, get_metrics_per_class
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
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # Initialize variables for tracking best model and training state
    best_val_loss = float('inf')
    best_model = build_model(num_classes=len(classes))
    cur_epoch = 1

    checkpoint_name = 'checkpoint.pth'
    checkpoint_path = os.path.join(get_config()['model_dir'], checkpoint_name)

    best_model_name = 'best_model.pth'
    best_model_path = os.path.join(get_config()['model_dir'], best_model_name)

    train_log_name = 'training_log.csv'

    # Check if checkpoint file exists and load it
    if Path(checkpoint_path).exists():
        cur_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_name)
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
    # Only train the model if final model file doesn't exist (to avoid overwriting best model if it exists)
    if not Path(best_model_path).exists():
        print(f"Starting training for {num_epochs - cur_epoch + 1} epochs on {device}...")
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
            if epoch > cur_epoch: # Only write log if it's a new epoch (not loaded from checkpoint)
                write_training_log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, file_name=train_log_name)
            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model.load_state_dict(model.state_dict())
                print(f"  ✓ Best model saved! (val_loss: {val_loss:.4f})")
                
            save_checkpoint(best_model, optimizer, epoch, best_val_loss, checkpoint_name)
            print("  ✓ Checkpoint saved!")

            gc.collect()
            torch.cuda.empty_cache()

        print("-" * 80)
        print("Training completed!")

    # Evaluate on test set with best model
    print("\nLoading best model and evaluating on test set...")
    _ = load_model(model, checkpoint_name, device)
    test_loss, test_acc = evaluate_model(model, device, test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    report, conf_matrix = get_metrics_per_class(model, device, test_dataloader, classes)
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    classification_report_name = 'classification_report.json'
    confusion_matrix_name = 'confusion_matrix.csv'
    save_classification_report(report, file_name=classification_report_name)
    save_confusion_matrix(conf_matrix, file_name=confusion_matrix_name)

    # Save the best model for future inference
    save_config = {
        'classes': classes,
        'image_size': config['image_size']
    }

    save_model(best_model, save_config, file_name=best_model_name)
    print(f"Best model saved to '{best_model_name}'")

if __name__ == '__main__':
    main()
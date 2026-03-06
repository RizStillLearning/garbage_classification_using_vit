# Garbage Classification using Vision Transformer

A simple Vision Transformer (ViT) based PyTorch project for classifying garbage images into multiple categories.

## ✅ Command-Line Usage

### 1) Train / Evaluate (end-to-end)
This project trains the model, evaluates it on the test set, and saves the best weights based on validation loss.

```bash
python main.py
```

> The training configuration (learning rate, batch size, number of epochs, etc.) is controlled via `config.yaml`.

### 2) Inference (Predict on a single image)
Once a model training is completed and saved to `best_model.pth`, run:

```bash
python predict.py path/to/image.jpg
```

Example:

```bash
python predict.py ./test_images/banana.jpg
```
Path to image can be either relative or absolute path.

## 🛠 Configuration

Modify `config.yaml` to tune hyperparameters and output paths:

- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training/validation
- `learning_rate`: Learning rate for optimizer
- `checkpoint_path`: Path where checkpoints are saved/loaded
- `best_model_path`: Final model output path
- `output_log_dir`: Training log CSV output
- `classification_report_dir`: Classification report path
- `confusion_matrix_dir`: Confusion matrix path

---

## 📂 Output Files

- `checkpoint.pth` – latest checkpoint during training
- `best_model.pth` – best model weights saved after training

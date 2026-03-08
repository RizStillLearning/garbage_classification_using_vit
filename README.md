# Garbage Classification using Vision Transformer

A simple Vision Transformer (ViT) based PyTorch project for classifying garbage images into multiple categories.

## ✅ Command-Line Usage

### 1) Train / Evaluate (end-to-end)
This project trains the model, evaluates it on the test set, and saves the best weights based on validation loss.

```bash
python main.py
```

> The training configuration (learning rate, batch size, number of epochs, etc.) is controlled via `config.yaml`.

> [!NOTE]
> Training can be paused anytime after the checkpoint is saved. When you resume the training, the model will simply load the latest checkpoint. If you wanna reset the whole training progress, delete the "checkpoint.pth" file. If the "best_model.pth" also exists, delete the file too.

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

### 3) Run the API Server
To run the API Server, you need to install uvicorn fia following command:
```bash
pip install uvicorn
```

Once the installation is done, run the API server with:
```bash
uvicorn app:app --reload
```

Here is the step by step to test the endpoint:
1. Open your web browser and go to http://127.0.0.1:8000/docs
2. Expand the POST `/predict/` bar and click `Try it out`
3. Select a garbage image from your files and click `Execute`

You will see a JSON response showing the predicted class and the confidence score like this:
```json
{
  "filename": "plastic.jpg",
  "predicted_class": "plastic",
  "probability": "1.00"
}
```

## 🛠 Configuration

Modify `config.yaml` to tune hyperparameters and output paths:

- `model_name`: Name of the pretrained model
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size for training/validation
- `learning_rate`: Learning rate for optimizer
- `weight_decay`: Weight decay
- `image_size`: Desired image size for preprocessing
- `model_dir`: Directory for model output
- `output_dir`: Directory for model training and evaluation log

---

## 📂 Output Files

- `checkpoint.pth` – latest checkpoint during training
- `best_model.pth` – best model weights saved after training
- `training_log.csv` - training log per epoch
- `classification_report.txt` - classification report (accuracy, precision, recall, f1-score)
- `confusion_matrix.csv` - confusion matrix (actual to predicted label matrix)

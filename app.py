import io
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from utils import get_device, get_transform, get_config
from model import build_model, load_model
from torch.nn import functional as F
# Initialize the FastAPI app
app = FastAPI(title="Garbage Classification API")

# Load the model and configuration at startup
config = get_config()
device = get_device()
model = build_model(num_classes=config['num_classes'])
save_config = load_model(model, file_name='best_model.pth', device=device)
classes = save_config['classes']

preprocess = get_transform('test')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = classes[predicted.item()]
            probability = F.softmax(outputs, dim=1)[0][predicted.item()].item()

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "probability": f"{probability:.2f}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")
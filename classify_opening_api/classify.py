from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
import uvicorn

# Import the model class and processing function
from train import DenseNNModel, process_fen

app = FastAPI(title="Chess Opening Classifier")

# Create Pydantic model for request
class FENRequest(BaseModel):
    fen: str

# Create Pydantic model for response
class OpeningPrediction(BaseModel):
    opening: str
    probability: float

# Global variables to store model and preprocessing objects
model = None
label_encoder = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    """Load the trained model and label encoder"""
    global model, label_encoder
    
    try:
        # Load label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Initialize model with the same architecture
        input_size = process_fen(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"])[0].shape[1]
        model = DenseNNModel(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            num_classes=len(label_encoder.classes_),
            dropout_rate=0.2
        )
        
        # Load trained weights
        model.load_state_dict(torch.load('models/dense_neural_network_v2.bin', map_location=device))
        model.to(device)
        model.eval()
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model and label encoder on startup"""
    if not load_model():
        raise RuntimeError("Failed to load model. Please ensure model files exist and are valid.")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "alive", "model_loaded": model is not None}

@app.post("/predict", response_model=OpeningPrediction)
async def predict_opening(request: FENRequest):
    """Predict chess opening from FEN string"""
    try:
        # Process input FEN
        X = process_fen([request.fen])
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Get predicted opening name
        predicted_opening = label_encoder.inverse_transform([predicted_idx])[0]
        
        return OpeningPrediction(
            opening=predicted_opening,
            probability=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("classify:app", host="0.0.0.0", port=8000, reload=True)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'dense_neural_network_v2.bin'
ENCODER_PATH = 'label_encoder.pkl'

# Create FastAPI app
app = FastAPI()

class FENRequest(BaseModel):
    fen: str

class OpeningPrediction(BaseModel):
    opening: str
    probability: float

class DenseNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(DenseNNModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_rate)])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def process_fen(fen_list):
    """Convert FEN strings to one-hot encoded vectors"""
    fen_strings = fen_list.to_numpy() if hasattr(fen_list, 'to_numpy') else fen_list

    vocab = sorted(set('rnbqkpRNBQKP12345678/- '))
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}

    max_length = max(len(fen.split(' ')[0]) for fen in fen_strings)
    num_samples = len(fen_strings)
    vocab_size = len(vocab)

    X = np.zeros((num_samples, max_length, vocab_size), dtype=np.float32)

    for i, fen in enumerate(fen_strings):
        board_part = fen.split(' ')[0]
        for j, char in enumerate(board_part):
            if char in char_to_idx:
                X[i, j, char_to_idx[char]] = 1.0

    return X.reshape(num_samples, -1)

def load_model():
    global model, label_encoder
    try:
        logger.info("Loading label encoder...")
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info("Loading model state...")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        input_size = state_dict['model.0.weight'].shape[1]
        model = DenseNNModel(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            num_classes=len(label_encoder.classes_),
            dropout_rate=0.2
        )
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError("Failed to load model")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/classify", response_model=OpeningPrediction)
async def predict_opening(request: FENRequest):
    try:
        X = process_fen([request.fen])
        if X.shape[1] != model.model[0].in_features:
            raise HTTPException(status_code=400, detail=f"Input size mismatch: expected {model.model[0].in_features}, got {X.shape[1]}")
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        return OpeningPrediction(opening=label_encoder.inverse_transform([predicted_idx])[0], probability=confidence)
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

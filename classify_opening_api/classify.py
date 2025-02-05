import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy: *ARRAY*API not found")

import os
import warnings
import sys

# Suppress specific numpy/torch warnings and stderr
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*NumPy v2.*")
os.environ['PYTHONNUMPY_API_WARNING'] = '0'

# Redirect stderr to devnull to suppress startup messages
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pickle
import logging

# Restore stderr after imports
sys.stderr = stderr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'dense_neural_network_v2.bin'
ENCODER_PATH = 'label_encoder.pkl'

# Initialize global variables
model = None
label_encoder = None
INPUT_SIZE = 1541  # Fixed input size from training

class FENRequest(BaseModel):
    fen: str

class OpeningClassification(BaseModel):
    opening: str
    certainty: float

class DenseNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.2):
        super(DenseNNModel, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def process_fen(fen_list):
    """Convert FEN strings to one-hot encoded vectors"""
    if isinstance(fen_list, str):
        fen_list = [fen_list]

    # Define vocabulary - MUST match training exactly
    vocab = sorted(set('rnbqkpRNBQKP12345678/- '))
    char_to_idx = {char: idx for idx, char in enumerate(vocab)}
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Initialize output array directly with final size
    num_samples = len(fen_list)
    X = np.zeros((num_samples, INPUT_SIZE), dtype=np.float32)
    
    # Process each FEN string
    for i, fen in enumerate(fen_list):
        board_part = fen.split(' ')[0]
        logger.info(f"Processing board part: {board_part}, length: {len(board_part)}")
        
        # One-hot encode each character
        for j, char in enumerate(board_part):
            if char in char_to_idx:
                # Calculate the correct position in the flattened array
                feature_idx = j * vocab_size + char_to_idx[char]
                if feature_idx < INPUT_SIZE:
                    X[i, feature_idx] = 1.0

    logger.info(f"Final shape: {X.shape}")
    return X

def load_model():
    """Load the model and label encoder"""
    global model, label_encoder
    try:
        logger.info("Loading label encoder...")
        with open(ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info("Loading model state...")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        
        model = DenseNNModel(
            input_size=INPUT_SIZE,
            hidden_sizes=[256, 128, 64],
            num_classes=len(label_encoder.classes_),
            dropout_rate=0.2
        )
        
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        logger.info(f"Model loaded successfully. Input size: {INPUT_SIZE}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the application...")
    load_model()
    yield
    logger.info("Shutting down the application...")
    global model, label_encoder
    model = None
    label_encoder = None

app = FastAPI(lifespan=lifespan)

@app.post("/classify", response_model=OpeningClassification)
async def predict_opening(request: FENRequest):
    try:
        # Process input
        logger.info(f"Processing FEN string: {request.fen}")
        X = process_fen(request.fen)
        logger.info(f"Processed features shape: {X.shape}")
        
        if X.shape[1] != INPUT_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Input size mismatch: got {X.shape[1]}, expected {INPUT_SIZE}"
            )
        
        # Make prediction
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get only the top prediction
            max_prob, max_idx = torch.max(probabilities[0], dim=0)
            
            # Convert to classification
            opening = label_encoder.inverse_transform([max_idx.item()])[0]
            classification = OpeningClassification(
                opening=opening,
                certainty=float(max_prob.item() * 100)  # Convert to percentage
            )
            
            return classification

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
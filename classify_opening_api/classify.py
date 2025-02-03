from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import uvicorn
import os
import sys
import traceback
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configurations
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'dense_neural_network_v2.bin'
ENCODER_PATH = 'label_encoder.pkl'

# Create FastAPI app
app = FastAPI(
    title="Chess Opening Classifier",
    description="API for classifying chess openings from FEN strings",
    version="1.0.0"
)

# Pydantic models for request/response
class FENRequest(BaseModel):
    fen: str

    class Config:
        json_schema_extra = {
            "example": {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
            }
        }

class OpeningPrediction(BaseModel):
    opening: str
    probability: float

# Neural Network Model Definition
class DenseNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout_rate: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def process_fen(fen_list: List[str]) -> np.ndarray:
    """Convert FEN strings to one-hot encoded vectors"""
    try:
        logger.info(f"Processing FEN list: {fen_list}")
        fen_strings = fen_list.to_numpy() if hasattr(fen_list, 'to_numpy') else fen_list
        logger.info(f"FEN strings after numpy conversion: {fen_strings}")

        vocab = sorted(set('rnbqkpRNBQKP12345678/- '))
        logger.info(f"Vocabulary size: {len(vocab)}")
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}

        # Debug the FEN string processing
        for fen in fen_strings:
            logger.info(f"Processing FEN: {fen}")
            board_part = fen.split(' ')[0]
            logger.info(f"Board part: {board_part}")

        max_length = max(len(fen.split(' ')[0]) for fen in fen_strings)
        logger.info(f"Max length: {max_length}")
        
        num_samples = len(fen_strings)
        vocab_size = len(vocab)
        logger.info(f"Creating array with shape: ({num_samples}, {max_length}, {vocab_size})")

        X = np.zeros((num_samples, max_length, vocab_size), dtype=np.float32)

        for i, fen in enumerate(fen_strings):
            board_part = fen.split(' ')[0]
            for j, char in enumerate(board_part):
                if char in char_to_idx:
                    X[i, j, char_to_idx[char]] = 1.0

        final_shape = X.reshape(num_samples, -1).shape
        logger.info(f"Final shape after reshape: {final_shape}")
        return X.reshape(num_samples, -1)
    except Exception as e:
        logger.error(f"Error in process_fen: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def load_model() -> bool:
    """Load the trained model and label encoder"""
    global model, label_encoder
    
    try:
        logger.info("Starting model loading process...")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Files in directory: {os.listdir()}")
        
        # Load label encoder
        logger.info("Loading label encoder...")
        if not os.path.exists('label_encoder.pkl'):
            raise FileNotFoundError("label_encoder.pkl not found")
            
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info(f"Label encoder loaded. Number of classes: {len(label_encoder.classes_)}")
        
        # Initialize model architecture
        logger.info("Initializing model architecture...")
        try:
            initial_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            logger.info(f"Using initial FEN: {initial_fen}")
            processed_fen = process_fen([initial_fen])
            input_size = processed_fen.shape[1]
            logger.info(f"Calculated input size: {input_size}")
        except Exception as e:
            logger.error(f"Error processing initial FEN: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        model = DenseNNModel(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            num_classes=len(label_encoder.classes_),
            dropout_rate=0.2
        )
        logger.info("Model architecture initialized")
        
        # Load model weights
        logger.info(f"Loading model weights using device: {DEVICE}")
        if not os.path.exists('dense_neural_network_v2.bin'):
            raise FileNotFoundError("dense_neural_network_v2.bin not found")
            
        state_dict = torch.load('dense_neural_network_v2.bin', map_location=DEVICE)
        logger.info("Model weights loaded")
        
        model.load_state_dict(state_dict)
        logger.info("Weights loaded into model")
        
        model.to(DEVICE)
        model.eval()
        logger.info("Model successfully loaded and set to eval mode")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False
    
@app.on_event("startup")
async def startup_event():
    """Initialize model and label encoder on startup"""
    if not load_model():
        raise RuntimeError("Failed to load model. Check logs for details.")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "alive",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }

@app.post("/classify", response_model=OpeningPrediction)
async def predict_opening(request: FENRequest):
    """
    Predict chess opening from FEN string
    
    Args:
        request: FENRequest object containing the FEN string
        
    Returns:
        OpeningPrediction: Predicted opening name and confidence score
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        logger.info(f"Processing FEN string: {request.fen}")
        
        # Process input FEN
        X = process_fen([request.fen])
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        # Get predicted opening name
        predicted_opening = label_encoder.inverse_transform([predicted_idx])[0]
        
        logger.info(f"Prediction successful. Opening: {predicted_opening}, Confidence: {confidence:.4f}")
        
        return OpeningPrediction(
            opening=predicted_opening,
            probability=confidence
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("classify:app", host="0.0.0.0", port=8000, reload=True)
# Chess Opening Classification

Access the GCP cloud run deployed live classification model via API at https://chess-opening-classification-api-akkxliqoya-uc.a.run.app/classify

## Problem Description
This project is a machine learning neural network solution to classify chess positions into their corresponding openings based on the board state (represented as a FEN codes). The model analyzes the current board position and classifies it into the Chess opening it thinks that this board most likely had. , providing the top 5 most likely classifications with their confidence scores. This is a multi-class classification problem where each board position can potentially match multiple standard chess openings, especially in cases of transpositions.

## Why work on this?
- Transposition Detection: Helps identify when different move orders lead to the same opening structure
- Game Analysis: Enables quick classification of games by opening type
- Learning Value: I figured this would be a fun and unique way to learn neural networks 

## Dataset

The original data used for this project is this 6,000,000 chess games datase from Kaggle thttps://www.kaggle.com/datasets/arevel/chess-game - These games were processed in order to generate ~2,000,000 board states to train and validate the model with. See the data/preprocessing folder for more information on how boards were extracted. 

### Key Dataset Features
- ECO (Encyclopedia of Chess Openings) codes
- Opening names
- Board positions (FEN notation)
- Turn numbers

Each record in the dataset maps a specific board position to its corresponding chess opening classification.

## Model Details

### Final Model

The final model implementation is a Dense neural network trained via Pytorch

I selected it because:
   - Optimal hyperparameter configuration
   - Balanced architecture size
   - Effective dropout rate
   - Appropriate learning rate
   - Best overall stability

### Model Performance and conclusions

## Model Performance Metrics

| Metric                | Model 1 (Dense) | Model 2 (Dense) | Model 3 (CNN) | Model 4 (CNN) |
|--------------------- |------------------|------------------|------------------|------------------|
| Final Val Accuracy   | ~13%             | ~13%             | ~13%             | ~12%             |
| Peak Val Accuracy    | ~13%             | ~13%             | ~18%             | ~12%             |
| Final Train Loss     | ~4.1             | ~4.8             | ~1.9             | ~4.9             |
| Final Val Loss       | ~5.1             | ~4.8             | ~9.2             | ~4.8             |
| Training Epochs      | 500              | 500              | 100              | 100              |
| Convergence Speed    | Medium (~150 epochs)| Medium (~100 epochs)| Fast (~10 epochs)| Fast (~20 epochs)|

## Model Parameters

| Parameter          | Model 1 (Dense)    | Model 2 (Dense)    | Model 3 (CNN)      | Model 4 (CNN)      |
|-------------------|-------------------|-------------------|-------------------|-------------------|
| Architecture      | Dense             | Dense             | CNN               | CNN               |
| Inner Layers      | [512, 256, 128]   | [256, 128, 64]    | Default           | Default           |
| Dropout Rate      | 0.0               | 0.2               | 0.0               | 0.3               |
| Learning Rate     | 0.0001            | 0.001             | 0.001             | 0.01              |
| Batch Size        | 4096              | 4096              | 4096              | 4096              |
| Epochs            | 500               | 500               | 100               | 100               |

## Model Characteristics

### Model 1 (Dense Neural Network)
- Large dense architecture with deeper layers [512, 256, 128]
- No dropout (0.0) which might explain minor overfitting
- Very conservative learning rate (0.0001)
- Shows minor signs of overfitting after epoch 200
- Stable validation accuracy
- Slower but steady learning due to small learning rate

### Model 2 (Dense Neural Network)
- Moderate dense architecture [256, 128, 64]
- Moderate dropout (0.2) helping prevent overfitting
- Balanced learning rate (0.001)
- Best overall stability and performance
- Optimal balance of regularization and model capacity

### Model 3 (Convolutional Neural Network)
- CNN architecture (layers not specified)
- No dropout (0.0) contributing to severe overfitting
- Moderate learning rate (0.001)
- Shows clear signs of overfitting
- Lack of regularization is problematic

### Model 4 (Convolutional Neural Network)
- CNN architecture (layers not specified)
- High dropout (0.3)
- Aggressive learning rate (0.01)
- Stable but suboptimal performance
- High learning rate might be preventing better convergence

## Conclusions

1. **Best Overall Model**: Model 2 (Dense)
   - Optimal hyperparameter configuration
   - Balanced architecture size
   - Effective dropout rate
   - Appropriate learning rate
   - Best overall stability

2. **Worst Overall Model**: Model 3 (CNN)
   - Lack of dropout leading to overfitting
   - CNN architecture may be unnecessarily complex
   - Would benefit from dropout and architecture specifications


## Project Structure
```
chess_opening_classifier/
├── classify_opening_api/
│   ├── classify.py
│   ├── dense_neural_network_v2.bin #Model files have been placed in this directory as well for ease of dockerizing
│   ├── dockerfile
│   ├── label_encoder.pkl
│   └── requirements.txt
├── data/
│   ├── preprocessing/
│   │   ├── process_boards.py
│   │   └── process_raw_csv.py
│   ├── .gitattributes
│   └── chess_boards_sample.csv
├── models/
│   ├── cnn_v1.bin
│   ├── cnn_v2.bin
│   ├── dense_neural_network_v1.bin
│   ├── dense_neural_network_v2.bin
│   └── label_encoder.pkl
├── .gitignore
├── LICENSE
├── notebook.ipynb
├── Pipfile
├── Pipfile.lock
├── README.md
├── test_api_cloud.py
├── test_api_local.py
└── train.py
```

##Environment Setup

Clone the repository:
```
git clone https://github.com/mar1-k/chess-opening-classifier.git
cd chess-opening-classifier
```

Create and activate a conda virtual environment:


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashCopypip install -r requirements.txt
Usage Instructions
Running the Notebook
bashCopyjupyter notebook notebook.ipynb
Training the Model
bashCopypython train.py
Running the Classification Service
bashCopypython predict.py
Using Docker
bashCopy# Build the Docker image
docker build -t chess-opening-classifier .

# Run the container
docker run -p 8000:8000 chess-opening-classifier
API Documentation
The classification service exposes the following endpoint:
POST /classify

Input: JSON with FEN string
Output: Top 5 most likely opening classifications with confidence scores

Example request:
jsonCopy{
    "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
}
Example response:
jsonCopy{
    "classifications": [
        {"opening": "King's Pawn Game", "confidence": 0.85},
        {"opening": "Italian Game", "confidence": 0.08},
        {"opening": "Ruy Lopez", "confidence": 0.04},
        {"opening": "Scotch Game", "confidence": 0.02},
        {"opening": "Vienna Game", "confidence": 0.01}
    ]
}
Model Performance
[To be completed after model training - will include:

Overall accuracy
Per-class metrics
Confusion matrix
Classification report]

Dependencies

Python 3.10+
polars
scikit-learn
numpy
flask
[other dependencies to be added]

Cloud Deployment
[To be completed after deployment - will include:

Deployment platform details
Access URL
Usage instructions
Monitoring and scaling information]

Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.
License
This project is licensed under the MIT License - see the LICENSE.md file for details.
Acknowledgments

Thanks to [data source] for providing the chess openings dataset
[Other acknowledgments]

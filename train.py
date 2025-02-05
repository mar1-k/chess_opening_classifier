import warnings
import os
import sys
from contextlib import redirect_stderr
from io import StringIO

# Silence specific warnings related to attempting to train on CPU
warnings.filterwarnings('ignore', category=UserWarning, module='torch.*')
warnings.filterwarnings('ignore', message='.*Numpy v2.*')

# Import torch with suppressed stderr
with redirect_stderr(StringIO()):
    import torch
    import torch.nn as nn
    import torch.optim as optim


import polars as pl
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

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

def train_chess_board_classifier(model, X_train, y_train, X_val, y_val, 
                     learning_rate, batch_size, epochs,
                     device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size].to(device)
            batch_y = y_train_tensor[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val_tensor[i:i+batch_size].to(device)
                batch_y = y_val_tensor[i:i+batch_size].to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_train_loss = total_train_loss / (len(X_train) / batch_size)
        avg_val_loss = total_val_loss / (len(X_val) / batch_size)
        val_acc = correct / total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")

    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")
    return history

def main():
    # Create models directory if it doesn't exist
    Path('./models').mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pl.read_csv('data/chess_boards_sample.csv')
    
    # Process features and target
    print("Processing data...")
    X = process_fen(df['Position'])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Opening'])
    
    # Create train/val split
    print("Creating train/val split...")
    n = len(df)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    
    np.random.seed(42)
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    X_train = X[idx[:n_train]]
    X_val = X[idx[n_train:n_train+n_val]]
    X_test = X[idx[n_train+n_val:]]
    
    y_train = y[idx[:n_train]]
    y_val = y[idx[n_train:n_train+n_val]]
    y_test = y[idx[n_train+n_val:]]
    
    # Model configuration
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    inner_layer = [256, 128, 64]
    
    # Training hyperparameters - YOU MAY HAVE TO ADJUST THESE BASED ON YOUR HARDWARE
    dropout_rate = 0.2
    learning_rate = 0.001
    batch_size = 4096
    epochs = 500
    
    print("\nTraining Configuration:")
    print(f"Dropout Rate: {dropout_rate}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Inner Layer: {inner_layer}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    
    # Initialize and train model
    print("\nInitializing model...")
    model = DenseNNModel(
        input_size=input_size,
        hidden_sizes=inner_layer,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    print("\nStarting training...")
    history = train_chess_board_classifier(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Save the model and label encoder
    print("\nSaving model and label encoder...")
    torch.save(model.state_dict(), './models/dense_neural_network_v2.bin')
    
    # Save label encoder
    with open('./models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
        
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
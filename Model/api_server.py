# api_server.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import re
import uvicorn
import os

# --- Configuration ---
MODEL_PATH = 'autocomplete_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ---------------------

app = FastAPI(title="RNN Autocomplete API")

# Setup CORS (Crucial for the browser HTML to talk to the server)
# Allows requests from any origin (*)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and metadata storage
global_data = {
    'model': None,
    'char_to_idx': None,
    'idx_to_char': None,
    'vocab_size': None,
    'seq_length': 15 # Default
}

# --- Request Body Definition ---
# FastAPI uses Pydantic models for request body validation
class PredictRequest(BaseModel):
    text: str
    max_length: int = 20
    temperature: float = 0.7

# --- Model Definition (Must be identical) ---
class AutocompleteRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super(AutocompleteRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.fc(output[:, -1, :]) 
        return output, hidden

# --- Helper Functions (Loaded into memory once) ---

def clean_text(text):
    # Must be the same cleaning logic used during training
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\'\-]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

def text_to_indices(text):
    # Uses the global char_to_idx loaded from the model file
    return [global_data['char_to_idx'].get(char, 0) for char in text.lower()]

def indices_to_text(indices):
    # Uses the global idx_to_char loaded from the model file
    return ''.join([global_data['idx_to_char'][idx] for idx in indices])


def autocomplete_working(start_text, max_length, temperature):
    """Generates the text using the loaded model."""
    model = global_data['model']
    seq_length = global_data['seq_length']
    
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    model.eval()
    start_indices = text_to_indices(start_text)
    
    # We only want the GENERATED text, not the input text, so we don't copy start_indices
    generated_indices = [] 

    with torch.no_grad():
        if len(start_indices) < seq_length:
            space_idx = global_data['char_to_idx'].get(' ', 0)
            # Pad the input for the first prediction
            input_sequence = [space_idx] * (seq_length - len(start_indices)) + start_indices
        else:
            input_sequence = start_indices[-seq_length:]
        
        current_seq = torch.tensor([input_sequence]).to(device)
        
        # We need to track the full history of the input + generated characters
        full_history = input_sequence.copy() 

        for i in range(max_length):
            # Get prediction
            output, _ = model(current_seq)
            output = output / temperature
            probabilities = torch.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probabilities, 1).item()

            # Add to generated indices and the history
            generated_indices.append(next_char_idx)
            full_history.append(next_char_idx)
            
            # Update sequence (sliding window)
            current_sequence = full_history[-seq_length:]
            current_seq = torch.tensor([current_sequence]).to(device)
            
            # Optional: stop if we generate punctuation
            if global_data['idx_to_char'][next_char_idx] in ['.', '!', '?'] and i > 5:
                break
    
    return indices_to_text(generated_indices)


# --- Startup Event (Load model when the server starts) ---
@app.on_event("startup")
async def startup_event():
    print("Loading model and metadata...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        
        # Load metadata
        global_data['char_to_idx'] = checkpoint['char_to_idx']
        global_data['idx_to_char'] = checkpoint['idx_to_char']
        global_data['vocab_size'] = checkpoint['vocab_size']
        global_data['seq_length'] = checkpoint.get('seq_length', 15)
        hidden_size = checkpoint['hidden_size']

        # Initialize and load model state
        model = AutocompleteRNN(global_data['vocab_size'], hidden_size).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        global_data['model'] = model
        
        print(f"Model loaded! Seq Length: {global_data['seq_length']}")
        
    except FileNotFoundError:
        print(f"FATAL: Model file not found at {MODEL_PATH}. Run train_and_save.py first!")
        # It's better to let the server start and fail on prediction than to crash startup
    except Exception as e:
        print(f"FATAL: An error occurred while loading the model: {e}")
        # Same here

# --- API Endpoint ---
@app.post("/predict")
async def predict_autocomplete(request_data: PredictRequest):
    """Endpoint to receive user text and return an autocomplete prediction."""
    
    # 1. Clean the input text
    cleaned_input = clean_text(request_data.text)
    
    if not cleaned_input:
        return {"prediction": ""}

    # 2. Get the prediction
    try:
        # The function now returns ONLY the generated characters
        generated_text = autocomplete_working(
            cleaned_input, 
            request_data.max_length, 
            request_data.temperature
        )
        
        return {"prediction": generated_text}
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# --- Execution Command ---
if __name__ == "__main__":
    # To run, execute 'python api_server.py' then visit http://127.0.0.1:8000
    # In practice, you'll use: uvicorn api_server:app --reload --host 0.0.0.0
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
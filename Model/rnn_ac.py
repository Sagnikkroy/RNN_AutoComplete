# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import re
from pathlib import Path

# %%
# --- Setup and Data Loading ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")
torch.manual_seed(42)
np.random.seed(42)


file_path = r'D:\RNN_AutoComplete\dataset\ds.txt' 


try:
    with open(file_path, 'r', encoding='utf-8') as file:
        training_text = file.read()
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path and try again.")
    exit()

print("training text:")
print(training_text[:500])  # Show first 500 characters

# %%
# --- Text Processing and Mapping Functions ---

def clean_text(text):
    # Remove unwanted characters and normalize whitespace
    # Using the same cleaning logic as before
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\'\-]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

# Clean the text first
training_text = clean_text(training_text)

# Get all unique characters from our cleaned text
chars = sorted(list(set(training_text.lower())))

# Create mappings between characters and numbers
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")

def text_to_indices(text):
    """Convert text string to list of numbers"""
    # Use .get() to handle characters not in the vocabulary by falling back to 0 (or a dedicated unknown token if defined)
    return [char_to_idx.get(char, 0) for char in text.lower()]

def indices_to_text(indices):
    """Convert list of numbers back to text"""
    return ''.join([idx_to_char[idx] for idx in indices])

# %%
# --- Data Sequence Creation ---

seq_length = 15  # Consistent sequence length

def create_sequences(text_indices, seq_length=15):
    """
    Create training examples where:
    - Input: sequence of characters
    - Output: next character in sequence
    """
    sequences = []
    next_chars = []
    
    for i in range(len(text_indices) - seq_length):
        seq = text_indices[i:i + seq_length]
        target = text_indices[i + seq_length]
        
        sequences.append(seq)
        next_chars.append(target)
    
    return torch.tensor(sequences), torch.tensor(next_chars)

# Convert our text to numbers and create training data
text_indices = text_to_indices(training_text)
X, y = create_sequences(text_indices, seq_length)

print(f"Input sequences shape: {X.shape}") 
print(f"Targets shape: {y.shape}") 

# %%
# --- Model Definition ---

class AutocompleteRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super(AutocompleteRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        
        # Output from the last time step only
        output = self.fc(output[:, -1, :]) 
        
        return output, hidden

# Model Initialization
hidden_size = 128
model = AutocompleteRNN(vocab_size, hidden_size).to(device)
print("Model created!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# --- Autocomplete Function (MOVED UP to fix NameError) ---

def autocomplete_working(model, start_text, max_length=30, temperature=0.8):
    """Working autocomplete function"""
    model.eval()
    
    # Convert to indices
    start_indices = text_to_indices(start_text)
    generated = start_indices.copy()
    
    with torch.no_grad():
        # Handle sequence length
        if len(start_indices) < seq_length:
            # Pad with space character (safe assumption)
            space_idx = char_to_idx.get(' ', 0) # Use .get for safety
            current_sequence = [space_idx] * (seq_length - len(start_indices)) + start_indices
        else:
            current_sequence = start_indices[-seq_length:]
        
        current_seq = torch.tensor([current_sequence]).to(device)
        
        for i in range(max_length):
            # Get prediction
            output, _ = model(current_seq)
            output = output / temperature
            
            # Convert to probabilities and sample (using multinomial for better control)
            probabilities = torch.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probabilities, 1).item()

            # Add to generated
            generated.append(next_char_idx)
            
            # Update sequence (sliding window)
            current_sequence = generated[-seq_length:]
            current_seq = torch.tensor([current_sequence]).to(device)
            
            # Optional: stop if we generate punctuation
            if idx_to_char[next_char_idx] in ['.', '!', '?'] and i > 5:
                break
    
    return indices_to_text(generated)


# %%
# --- Training Loop ---

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Create DataLoader for batching
batch_size = 64
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, dataloader, epochs=50):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

            # Test autocomplete after every 1 epochs (uses the now-defined function)
            model.eval()
            test_input = "hello"
            with torch.no_grad():
                # The call to autocomplete_working is now valid
                completion = autocomplete_working(model, test_input, max_length=10, temperature=0.7)
            print(f"  Test: '{test_input}' -> '{completion}'")
            model.train()
    
    return losses

# Train the model
print("Starting training...")
losses = train_model(model, dataloader, epochs=200)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# %%


# %%
# Save the trained model
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
    'hidden_size': hidden_size
}, 'autocomplete_model.pth')

print("Model saved as 'autocomplete_model.pth'")
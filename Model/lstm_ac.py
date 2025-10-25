# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import re

# %%
# --- Setup and Data Loading ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

file_path = r'D:\RNN_AutoComplete\dataset\ds.txt' 

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        training_text = file.read()
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path and try again.")
    exit()

print("Training text preview:")
print(training_text[:500])

# %%
# --- Text Processing and Mapping Functions ---

def clean_text(text):
    """Remove unwanted characters and normalize whitespace"""
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\'\-]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

# Clean the text
training_text = clean_text(training_text)

# Get all unique characters
chars = sorted(list(set(training_text.lower())))

# Create character mappings
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}
vocab_size = len(chars)

print(f"\nVocabulary size: {vocab_size}")
print(f"Characters: {chars}")

def text_to_indices(text):
    """Convert text string to list of indices"""
    return [char_to_idx.get(char, 0) for char in text.lower()]

def indices_to_text(indices):
    """Convert list of indices back to text"""
    return ''.join([idx_to_char[idx] for idx in indices])

# %%
# --- Data Sequence Creation ---

seq_length = 25  # Length of input sequences

def create_sequences(text_indices, seq_length):
    """Create training sequences"""
    sequences = []
    next_chars = []
    
    for i in range(len(text_indices) - seq_length):
        seq = text_indices[i:i + seq_length]
        target = text_indices[i + seq_length]
        sequences.append(seq)
        next_chars.append(target)
    
    return torch.tensor(sequences, dtype=torch.long), torch.tensor(next_chars, dtype=torch.long)

# Convert text to indices and create training data
text_indices = text_to_indices(training_text)
X, y = create_sequences(text_indices, seq_length)

print(f"\nInput sequences shape: {X.shape}")
print(f"Targets shape: {y.shape}")
print(f"Total training samples: {len(X)}")

# %%
# --- Model Definition ---

class AutocompleteRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.3):
        super(AutocompleteRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM forward pass
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        
        # Use only the last time step output
        output = output[:, -1, :]  # (batch_size, hidden_size)
        output = self.dropout(output)
        output = self.fc(output)  # (batch_size, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
                  weight.new_zeros(self.num_layers, batch_size, self.hidden_size))
        return hidden

# Model initialization
embedding_dim = 64
hidden_size = 256
num_layers = 2

model = AutocompleteRNN(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=0.3
).to(device)

print("\nModel created!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# --- Autocomplete Function ---

def autocomplete(model, start_text, max_length=50, temperature=0.8):
    """
    Generate autocomplete predictions
    
    Args:
        model: trained RNN model
        start_text: starting text prompt
        max_length: maximum characters to generate
        temperature: sampling temperature (higher = more random)
    """
    model.eval()
    
    # Convert start text to indices
    start_indices = text_to_indices(start_text)
    
    if len(start_indices) == 0:
        print("Warning: Empty start text, using space")
        start_indices = [char_to_idx.get(' ', 0)]
    
    generated = start_indices.copy()
    
    with torch.no_grad():
        # Prepare initial sequence
        if len(start_indices) < seq_length:
            # Pad with spaces
            space_idx = char_to_idx.get(' ', 0)
            current_sequence = [space_idx] * (seq_length - len(start_indices)) + start_indices
        else:
            current_sequence = start_indices[-seq_length:]
        
        hidden = None
        
        # Generate characters one by one
        for i in range(max_length):
            # Prepare input
            current_seq = torch.tensor([current_sequence], dtype=torch.long).to(device)
            
            # Get prediction
            output, hidden = model(current_seq, hidden)
            
            # Apply temperature
            output = output / temperature
            
            # Sample from distribution
            probabilities = torch.softmax(output, dim=-1)
            next_char_idx = torch.multinomial(probabilities[0], 1).item()
            
            # Add to generated sequence
            generated.append(next_char_idx)
            
            # Update sequence (sliding window)
            current_sequence = current_sequence[1:] + [next_char_idx]
            
            # Optional: stop at sentence endings
            if idx_to_char[next_char_idx] in ['.', '!', '?'] and i > 10:
                break
    
    return indices_to_text(generated)

# %%
# --- Training Setup ---

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Create DataLoader
batch_size = 128
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"\nBatches per epoch: {len(dataloader)}")

# %%
# --- Training Loop ---

def train_model(model, dataloader, epochs=150, test_every=5):
    """Train the RNN model"""
    model.train()
    losses = []
    
    print("\nStarting training...\n")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Print progress
        if (epoch + 1) % test_every == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            # Test autocomplete
            model.eval()
            test_prompts = ["the", "hello", "i am"]
            
            with torch.no_grad():
                for prompt in test_prompts:
                    completion = autocomplete(model, prompt, max_length=30, temperature=0.7)
                    print(f"  '{prompt}' -> '{completion}'")
            
            print()
            model.train()
    
    return losses

# Train the model
losses = train_model(model, dataloader, epochs=1000, test_every=5)

# %%
# --- Plot Training Loss ---

plt.figure(figsize=(12, 5))
plt.plot(losses, linewidth=2)
plt.title('Training Loss Over Time', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# --- Test Autocomplete ---

print("\n" + "="*50)
print("Testing Autocomplete")
print("="*50 + "\n")

model.eval()
test_inputs = [
    "the quick",
    "hello",
    "i think",
    "what is",
    "once upon"
]

for test_input in test_inputs:
    with torch.no_grad():
        completion = autocomplete(model, test_input, max_length=40, temperature=0.7)
        print(f"Input: '{test_input}'")
        print(f"Output: '{completion}'")
        print()

# %%
# --- Save Model ---

save_dict = {
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
    'embedding_dim': embedding_dim,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'seq_length': seq_length
}

torch.save(save_dict, 'autocomplete_model_lstm.pth')
print("Model saved as 'autocomplete_model_lstm.pth'")

# %%
# --- Load Model (for future use) ---

def load_model(filepath):
    """Load a saved model"""
    checkpoint = torch.load(filepath, map_location=device)
    
    loaded_model = AutocompleteRNN(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    ).to(device)
    
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    return loaded_model, checkpoint

# Example: model, checkpoint = load_model('autocomplete_model_lstm.pth')
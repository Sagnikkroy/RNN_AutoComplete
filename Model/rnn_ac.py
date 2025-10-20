# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")
torch.manual_seed(42)
np.random.seed(42)

# %%
with open('ds.txt', 'r') as file:
    training_text = file.read()

print("training text:")
print(training_text)

# %%
# Get all unique characters from our text
chars = sorted(list(set(training_text.lower())))
print(f"Unique characters: {chars}")
print(f"Total unique characters: {len(chars)}")

# Create mappings between characters and numbers
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

print("\nCharacter to index mapping:")
for char, idx in list(char_to_idx.items())[:10]:  # Show first 10
    print(f"  '{char}' -> {idx}")

vocab_size = len(chars)
print(f"\nVocabulary size: {vocab_size}")

# %%
def text_to_indices(text):
    """Convert text string to list of numbers"""
    return [char_to_idx[char] for char in text.lower()]

def indices_to_text(indices):
    """Convert list of numbers back to text"""
    return ''.join([idx_to_char[idx] for idx in indices])

# Let's test our conversion
test_text = "hello"
test_indices = text_to_indices(test_text)
converted_back = indices_to_text(test_indices)

print(f"Original: '{test_text}'")
print(f"To indices: {test_indices}")
print(f"Back to text: '{converted_back}'")

# %%
def create_sequences(text_indices, seq_length=20):
    """
    Create training examples where:
    - Input: sequence of characters
    - Output: next character in sequence
    """
    sequences = []
    next_chars = []
    
    # Slide a window through the text
    for i in range(len(text_indices) - seq_length):
        # Input sequence
        seq = text_indices[i:i + seq_length]
        # Target (next character after the sequence)
        target = text_indices[i + seq_length]
        
        sequences.append(seq)
        next_chars.append(target)
    
    return torch.tensor(sequences), torch.tensor(next_chars)

# Convert our text to numbers
text_indices = text_to_indices(training_text)

# Create training data
seq_length = 15  # We'll use 15 characters to predict the 16th
X, y = create_sequences(text_indices, seq_length)

print(f"Input sequences shape: {X.shape}")  # (num_sequences, seq_length)
print(f"Targets shape: {y.shape}")         # (num_sequences,)

# Let's look at one example
print(f"\nFirst training example:")
input_seq = X[0]
target_char = y[0]
print(f"Input: '{indices_to_text(input_seq.tolist())}'")
print(f"Target: '{idx_to_char[target_char.item()]}'")

# %%
class AutocompleteRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super(AutocompleteRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer: converts character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # GRU layer: our RNN that remembers patterns
        self.gru = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer: predicts which character comes next
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length)
        
        # Step 1: Convert character indices to vectors
        embedded = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        
        # Step 2: Pass through GRU (the RNN part)
        output, hidden = self.gru(embedded, hidden)
        
        # Step 3: Get the last output and predict next character
        output = self.fc(output[:, -1, :])  # Use only the last output
        
        return output, hidden

# Let's create our model
hidden_size = 128
model = AutocompleteRNN(vocab_size, hidden_size).to(device)
print("Model created!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
def autocomplete(model, start_text, max_length=50, temperature=0.8, seq_length=15):
    """Generate autocomplete suggestions - IMPROVED VERSION"""
    model.eval()
    
    # Convert start text to indices
    start_indices = text_to_indices(start_text)
    
    # Handle short inputs by padding
    if len(start_indices) < seq_length:
        # We can either pad or use what we have
        # Let's use what we have but warn the user
        print(f"Warning: Input '{start_text}' is shorter than sequence length {seq_length}")
        # We'll just use the available characters
        current_sequence = start_indices
    else:
        # Use the last seq_length characters
        current_sequence = start_indices[-seq_length:]
    
    generated = start_indices.copy()
    
    with torch.no_grad():
        # Convert to tensor with correct shape: (batch_size=1, sequence_length)
        current_seq = torch.tensor([current_sequence]).to(device)
        print(f"Starting with sequence: '{indices_to_text(current_sequence)}'")
        
        for i in range(max_length):
            # Forward pass
            output, _ = model(current_seq)
            
            # Apply temperature
            output = output / temperature
            
            # Get probabilities
            probabilities = torch.softmax(output, dim=-1).cpu().numpy()[0]
            
            # Sample next character
            next_char_idx = np.random.choice(len(probabilities), p=probabilities)
            generated.append(next_char_idx)
            
            # Update sequence (sliding window)
            new_sequence = generated[-seq_length:]
            current_seq = torch.tensor([new_sequence]).to(device)
            
            # Optional: stop if we generate a newline or similar
            if idx_to_char[next_char_idx] == '\n':
                break
    
    final_text = indices_to_text(generated)
    print(f"Final result: '{final_text}'")
    return final_text

# %%
def autocomplete_working(model, start_text, max_length=30, temperature=0.8):
    """Working autocomplete function"""
    model.eval()
    seq_length = 15
    
    # Convert to indices
    start_indices = text_to_indices(start_text)
    generated = start_indices.copy()
    
    with torch.no_grad():
        # Handle sequence length
        if len(start_indices) < seq_length:
            # Pad with zeros (0 is usually space or most common char)
            current_sequence = [0] * (seq_length - len(start_indices)) + start_indices
        else:
            current_sequence = start_indices[-seq_length:]
        
        current_seq = torch.tensor([current_sequence]).to(device)
        
        for i in range(max_length):
            # Get prediction
            output, _ = model(current_seq)
            output = output / temperature
            
            # Convert to probabilities and sample
            probabilities = torch.softmax(output, dim=-1).cpu().numpy()[0]
            next_char_idx = np.random.choice(len(probabilities), p=probabilities)
            
            # Add to generated
            generated.append(next_char_idx)
            
            # Update sequence (sliding window)
            current_sequence = generated[-seq_length:]
            current_seq = torch.tensor([current_sequence]).to(device)
    
    return indices_to_text(generated)

# Test it!
print("\n=== Testing Fixed Autocomplete ===")
test_inputs = ["hello", "mach", "neur", "pyt"]

for test_input in test_inputs:
    completion = autocomplete_working(model, test_input, max_length=20, temperature=0.7)
    print(f"Input: '{test_input}' -> '{completion}'")

# %%
def interactive_demo_fixed():
    print("\n🎯 === INTERACTIVE AUTOCOMPLETE (FIXED) ===")
    print("Type some text and see what the model suggests!")
    
    while True:
        user_input = input("\nStart typing: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif len(user_input) < 1:
            print("Please type at least one character")
            continue
        
        completion = autocomplete_working(model, user_input, max_length=30, temperature=0.7)
        
        # Show the original input and the completion in different colors
        original_part = completion[:len(user_input)]
        new_part = completion[len(user_input):]
        print(f"You: {original_part}\033[94m{new_part}\033[0m")

# Run the fixed demo
interactive_demo_fixed()



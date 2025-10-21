# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import re

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using {device}")
torch.manual_seed(42)
np.random.seed(42)

# %%
with open('D:\\RNN_AutoComplete\\dataset\\ds.txt', 'r') as file:
    training_text = file.read()

print("training text:")
print(training_text[:500])  # Show first 500 characters

# %%
def clean_text(text):
    # Remove unwanted characters and normalize whitespace
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\'\-]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned

# Clean the text first
training_text = clean_text(training_text)

# Get all unique characters from our cleaned text
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
# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoader for batching
batch_size = 64
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
# Training loop
def train_model(model, dataloader, epochs=50):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            # Move data to device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
            
            # Test autocomplete after every 10 epochs
            model.eval()
            test_input = "hello"
            with torch.no_grad():
                completion = autocomplete_working(model, test_input, max_length=10, temperature=0.7)
            print(f"  Test: '{test_input}' -> '{completion}'")
            model.train()
    
    return losses

# Train the model
print("Starting training...")
losses = train_model(model, dataloader, epochs=100)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

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
            # Pad with space character (find space index)
            space_idx = char_to_idx[' ']
            current_sequence = [space_idx] * (seq_length - len(start_indices)) + start_indices
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
            
            # Optional: stop if we generate punctuation that might end a sentence
            if idx_to_char[next_char_idx] in ['.', '!', '?', '\n'] and i > 5:
                break
    
    return indices_to_text(generated)

# %%
# Test the trained model
print("\n=== Testing Trained Autocomplete ===")
test_inputs = ["hello", "mach", "neur", "pyt", "artificial", "the quick"]

for test_input in test_inputs:
    completion = autocomplete_working(model, test_input, max_length=20, temperature=0.7)
    print(f"Input: '{test_input}' -> '{completion}'")

# %%
def interactive_demo_fixed():
    print("Type some text and see what the model suggests!")
    print("Type 'quit' to exit.")
    
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
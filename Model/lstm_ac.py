# lstm_ac.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import re
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --------------------------------------------------------------
# 1. FORCE CPU TO AVOID GPU OOM
# --------------------------------------------------------------
device = torch.device('cpu')
print("FORCING CPU TO AVOID GPU OOM")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------------------------------------------------
# 2. LOAD TEXT
# --------------------------------------------------------------
file_path = r'dataset\ds.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    training_text = f.read()

print("\nTraining text preview:")
print(training_text[:500])

def clean_text(text: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\'\-]', '', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

training_text = clean_text(training_text)

# --------------------------------------------------------------
# 3. VOCABULARY
# --------------------------------------------------------------
chars = sorted(list(set(training_text.lower())))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)
print(f"\nVocabulary size: {vocab_size}")

def text_to_indices(text: str):
    return [char_to_idx.get(c, 0) for c in text.lower()]

def indices_to_text(indices):
    return ''.join(idx_to_char.get(i, '') for i in indices)

# --------------------------------------------------------------
# 4. SEQUENCES
# --------------------------------------------------------------
seq_length = 25
text_indices = text_to_indices(training_text)

X, y = [], []
for i in range(len(text_indices) - seq_length):
    X.append(text_indices[i:i + seq_length])
    y.append(text_indices[i + seq_length])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
print(f"Sequences: {X.shape}, Targets: {y.shape}")

# --------------------------------------------------------------
# 5. MODEL
# --------------------------------------------------------------
class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_size=256, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out, hidden

model = TinyLSTM(vocab_size, embed_dim=32, hidden_size=64, num_layers=1)
model = model.to(device)  # ← Now safe on CPU
print(f"Model on {device}! Parameters: {sum(p.numel() for p in model.parameters())}")

# --------------------------------------------------------------
# 6. DATALOADER
# --------------------------------------------------------------
batch_size = 32  # ← Can be larger on CPU
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# --------------------------------------------------------------
# 7. AUTOCOMPLETE
# --------------------------------------------------------------
def autocomplete(model, start_text, max_length=50, temperature=0.8):
    model.eval()
    idx = text_to_indices(start_text)
    if not idx:
        idx = [char_to_idx.get(' ', 0)]
    generated = idx.copy()

    if len(idx) < seq_length:
        seq = [char_to_idx.get(' ', 0)] * (seq_length - len(idx)) + idx
    else:
        seq = idx[-seq_length:]

    hidden = None
    with torch.no_grad():
        for _ in range(max_length):
            inp = torch.tensor([seq], dtype=torch.long).to(device)
            logits, hidden = model(inp, hidden)
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            generated.append(nxt)
            seq = seq[1:] + [nxt]
            if idx_to_char[nxt] in '.!?' and len(generated) > 10:
                break
    return indices_to_text(generated)

# --------------------------------------------------------------
# 8. TRAINING
# --------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

def train(epochs=50, log_every=5):
    model.train()
    losses = []
    print("\n=== TRAINING START ===\n")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                    desc=f"Epoch {epoch}", leave=False)

        for batch_idx, (xb, yb) in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = epoch_loss / len(dataloader)
        losses.append(avg)
        scheduler.step(avg)

        if epoch % log_every == 0:
            print(f"\nEpoch {epoch} | Loss: {avg:.4f}")
            model.eval()
            for p in ["the", "hello", "python"]:
                print(f"  '{p}' → '{autocomplete(model, p, 30, 0.7)}'")
            model.train()

    return losses

# --------------------------------------------------------------
# 9. RUN
# --------------------------------------------------------------
losses = train(epochs=50, log_every=5)

# Plot
plt.plot(losses)
plt.title("Training Loss (CPU)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Final test
print("\nFINAL OUTPUT:")
for p in ["the quick", "hello", "what is ai"]:
    print(f"{p} → {autocomplete(model, p, 40, 0.7)}")

# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
    'seq_length': seq_length
}, 'final_model_cpu.pth')
print("Model saved!")
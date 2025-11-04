# lstm_ac.py
# Full LSTM autocomplete trainer
# Saves model compatible with api_server.py

import os
import re
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ========================================
# 1. CONFIGURATION
# ========================================
SEQ_LENGTH   = 25
BATCH_SIZE   = 64
EMBED_DIM    = 32
HIDDEN_SIZE  = 256        # ← MUST MATCH API
NUM_LAYERS   = 1
EPOCHS       = 50
LR           = 0.002
DEVICE       = torch.device('cpu')  # Change to 'cuda' if you have GPU
SAVE_PATH    = "final_model_cpu.pth"  # ← This is your LSTM model

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ========================================
# 2. LOAD TEXT
# ========================================
FILE_PATH = r'dataset\ds.txt'

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"{FILE_PATH} not found!")

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    raw_text = f.read()

print("\n--- Text preview (first 500 chars) ---")
print(raw_text[:500])
print("...")

def clean_text(text: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\'\-]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip()

text = clean_text(raw_text)

# ========================================
# 3. VOCABULARY
# ========================================
chars = sorted(list(set(text.lower())))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)
print(f"\nVocabulary size: {vocab_size}")

def text_to_indices(s: str):
    return [char_to_idx.get(c, 0) for c in s.lower()]

def indices_to_text(idx):
    return ''.join(idx_to_char.get(i, '') for i in idx)

# ========================================
# 4. SEQUENCES
# ========================================
indices = text_to_indices(text)
X, y = [], []
for i in range(len(indices) - SEQ_LENGTH):
    X.append(indices[i:i + SEQ_LENGTH])
    y.append(indices[i + SEQ_LENGTH])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
print(f"X: {X.shape} | y: {y.shape}")

# ========================================
# 5. DATALOADER
# ========================================
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ========================================
# 6. MODEL
# ========================================
class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
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

model = TinyLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)
print(f"Model on {DEVICE} | Params: {sum(p.numel() for p in model.parameters()):,}")

# ========================================
# 7. AUTOCOMPLETE
# ========================================
def autocomplete(start_text: str, max_len=50, temp=0.8):
    model.eval()
    idx = text_to_indices(start_text)
    generated = idx[:]
    seq = idx[-SEQ_LENGTH:] if len(idx) >= SEQ_LENGTH else [char_to_idx.get(' ', 0)] * (SEQ_LENGTH - len(idx)) + idx
    hidden = None
    with torch.no_grad():
        for _ in range(max_len):
            inp = torch.tensor([seq], dtype=torch.long).to(DEVICE)
            logits, hidden = model(inp, hidden)
            logits = logits / temp
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            generated.append(nxt)
            seq = seq[1:] + [nxt]
            if idx_to_char[nxt] in '.!?' and len(generated) > 10:
                break
    return indices_to_text(generated)

# ========================================
# 8. TRAINING
# ========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

def train_one_epoch():
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred, _ = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

print("\n=== START TRAINING ===\n")
loss_history = []

for epoch in range(1, EPOCHS + 1):
    avg_loss = train_one_epoch()
    scheduler.step(avg_loss)
    loss_history.append(avg_loss)
    if epoch % 5 == 0:
        print(f"\nEpoch {epoch:02d} | Loss: {avg_loss:.4f}")
        for seed in ["the", "hello", "python"]:
            print(f"  '{seed}' → '{autocomplete(seed, 30, 0.7)}'")

# ========================================
# 9. PLOT
# ========================================
plt.figure(figsize=(8, 4))
plt.plot(loss_history, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ========================================
# 10. FINAL DEMO
# ========================================
print("\n=== FINAL AUTOCOMPLETE ===")
for seed in ["the quick", "hello", "what is your"]:
    print(f"{seed} → {autocomplete(seed, 40, 0.7)}")

# ========================================
# 11. SAVE MODEL – COMPATIBLE WITH API
# ========================================
torch.save({
    'model_state_dict': model.state_dict(),
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
    'seq_length': SEQ_LENGTH,
    'embed_dim': EMBED_DIM,
    'hidden_size': HIDDEN_SIZE,      # ← CRITICAL
    'num_layers': NUM_LAYERS,
}, SAVE_PATH)

print(f"\nModel saved to: {SAVE_PATH}")
print("Now run: python api_server.py")
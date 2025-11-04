# api_server.py
# UNIVERSAL DUAL MODEL SERVER
# Works with ANY .pth file (GRU/LSTM, any num_layers, any embed_dim, any hidden_size)
# NO RETRAIN NEEDED

import os
import re
import threading
import uvicorn
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# ========================================
# CONFIG – UPDATE PATHS ONLY
# ========================================
MODEL_CONFIGS = [
    {
        "path": "autocomplete_model.pth",       # ← Your GRU model (2 layers)
        "port": 8000,
        "title": "GRU Autocomplete (8000)"
    },
    {
        "path": "final_model_cpu.pth",          # ← Your LSTM model (1 layer)
        "port": 8001,
        "title": "LSTM Autocomplete (8001)"
    }
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# UNIVERSAL DYNAMIC RNN
# ========================================
class UniversalRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, rnn_type="gru"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers,
                               batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_size, num_layers,
                              batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

# ========================================
# REQUEST
# ========================================
class PredictRequest(BaseModel):
    text: str
    max_length: int = 20
    temperature: float = 0.7

# ========================================
# STATE DICT CLEANER (removes unexpected layers)
# ========================================
def clean_state_dict(state_dict, expected_layers):
    cleaned = {}
    for k, v in state_dict.items():
        # Only keep layer 0 to expected_layers-1
        if "rnn.weight_ih_l" in k or "rnn.weight_hh_l" in k or "rnn.bias" in k:
            layer_num = int(k.split("_l")[-1][0]) if "_l" in k else 0
            if layer_num < expected_layers:
                cleaned[k] = v
        else:
            cleaned[k] = v
    return cleaned

# ========================================
# APP FACTORY
# ========================================
def create_app(cfg: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title=cfg["title"])
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    data = {
        "model": None, "char_to_idx": None, "idx_to_char": None,
        "vocab_size": None, "seq_length": 15, "rnn_type": "gru"
    }

    def clean_text(text: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\'\-]', '', text)).strip()

    def text_to_indices(text: str):
        return [data["char_to_idx"].get(c, 0) for c in text.lower()]

    def indices_to_text(indices):
        return "".join(data["idx_to_char"].get(i, "") for i in indices)

    def autocomplete(start_text: str, max_length: int, temperature: float) -> str:
        if not data["model"]:
            raise HTTPException(500, "Model not loaded")
        model, seq_len = data["model"], data["seq_length"]
        model.eval()
        indices = text_to_indices(start_text)
        generated = []

        with torch.no_grad():
            if len(indices) < seq_len:
                pad = [data["char_to_idx"].get(" ", 0)] * (seq_len - len(indices))
                seq = torch.tensor([pad + indices], dtype=torch.long).to(DEVICE)
            else:
                seq = torch.tensor([indices[-seq_len:]], dtype=torch.long).to(DEVICE)

            history = seq.squeeze(0).tolist()

            for _ in range(max_length):
                out, _ = model(seq)
                out = out / temperature
                probs = torch.softmax(out, dim=-1)
                nxt = torch.multinomial(probs, 1).item()
                generated.append(nxt)
                history.append(nxt)
                seq = torch.tensor([history[-seq_len:]], dtype=torch.long).to(DEVICE)
                if data["idx_to_char"].get(nxt) in ".!?" and len(generated) > 5:
                    break

        return indices_to_text(generated)

    @app.on_event("startup")
    async def load_model():
        path, port = cfg["path"], cfg["port"]
        print(f"[{port}] Loading {path}...")
        try:
            ckpt = torch.load(path, map_location=DEVICE)

            # Detect RNN type
            state_keys = list(ckpt["model_state_dict"].keys())
            rnn_type = "lstm" if any("lstm." in k for k in state_keys) else "gru"

            # Get architecture from checkpoint
            embed_dim = ckpt.get("embed_dim", ckpt.get("hidden_size", 128))
            hidden_size = ckpt.get("hidden_size", 128)

            # Count layers in state_dict
            max_layer = 0
            for k in state_keys:
                if "weight_ih_l" in k or "weight_hh_l" in k:
                    try:
                        layer_num = int(k.split("_l")[-1][0])
                        max_layer = max(max_layer, layer_num + 1)
                    except:
                        pass
            num_layers = max_layer if max_layer > 0 else ckpt.get("num_layers", 1)

            data.update({
                "char_to_idx": ckpt["char_to_idx"],
                "idx_to_char": ckpt["idx_to_char"],
                "vocab_size": ckpt["vocab_size"],
                "seq_length": ckpt.get("seq_length", 25),
                "rnn_type": rnn_type
            })

            # Create model with EXACT architecture
            model = UniversalRNN(
                vocab_size=data["vocab_size"],
                embed_dim=embed_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                rnn_type=rnn_type
            ).to(DEVICE)

            # Clean state dict to match model
            clean_sd = clean_state_dict(ckpt["model_state_dict"], num_layers)
            # Remap gru./lstm. → rnn.
            remapped = {}
            for k, v in clean_sd.items():
                if k.startswith("gru.") or k.startswith("lstm."):
                    k = k.replace("gru.", "rnn.").replace("lstm.", "rnn.")
                remapped[k] = v

            model.load_state_dict(remapped, strict=True)
            model.eval()
            data["model"] = model

            print(f"[{port}] Loaded! {rnn_type.upper()}, "
                  f"embed={embed_dim}, hidden={hidden_size}, layers={num_layers}, vocab={data['vocab_size']}")

        except Exception as e:
            print(f"[{port}] FAILED: {e}")
            import traceback
            traceback.print_exc()

    @app.post("/predict")
    async def predict(req: PredictRequest):
        cleaned = clean_text(req.text)
        if not cleaned:
            return {"prediction": ""}
        try:
            return {"prediction": autocomplete(cleaned, req.max_length, req.temperature)}
        except Exception as e:
            raise HTTPException(500, f"Error: {e}")

    return app

# ========================================
# RUN
# ========================================
def run_server(cfg):
    uvicorn.run(create_app(cfg), host="0.0.0.0", port=cfg["port"])

if __name__ == "__main__":
    print("Starting universal dual-model server...")
    threads = []
    for cfg in MODEL_CONFIGS:
        t = threading.Thread(target=run_server, args=(cfg,), daemon=True)
        t.start()
        threads.append(t)

    print("Servers: http://127.0.0.1:8000 (GRU) | http://127.0.0.1:8001 (LSTM)")
    try:
        for t in threads: t.join()
    except KeyboardInterrupt:
        print("\nStopped.")
# resave_charcnn.py
import os
import json
import torch

from src.train_char_cnn import CharCNN   # your model class

MODEL_DIR = "models"

model_path = os.path.join(MODEL_DIR, "charcnn_model.pt")
vocab_path = os.path.join(MODEL_DIR, "charcnn_vocab.json")
cfg_path = os.path.join(MODEL_DIR, "charcnn_config.json")

# -------------------------
# Load vocab + config
# -------------------------
print("Loading vocab + config…")

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

# -------------------------
# Load old checkpoint
# -------------------------
print("Loading saved checkpoint…")

checkpoint = torch.load(model_path, map_location="cpu")

# Detect which form it is
if "model_state" in checkpoint:
    print("✔ Found wrapped checkpoint (model_state + metadata)")
    state_dict = checkpoint["model_state"]
else:
    print("✔ Found raw state_dict")
    state_dict = checkpoint

# -------------------------
# Rebuild model
# -------------------------
model = CharCNN(vocab_size=cfg["vocab_size"])

# Load only the pure state_dict
model.load_state_dict(state_dict)

# -------------------------
# Re-save in clean format
# -------------------------
print("Re-saving model in safe zip format…")

torch.save(
    model.state_dict(),
    model_path,
    _use_new_zipfile_serialization=True
)

print("🎉 SUCCESS — charcnn_model.pt is now safe and streamlit-compatible!")

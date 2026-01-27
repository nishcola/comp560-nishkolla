"""
Prepare the "Cyclic Alphabet" dataset.
Task: "abcdef...z" repeated.
Rationale: Zero variance, local dependency only. 
If the model fails this, the architecture is broken.
"""
import os
import pickle
import string
import numpy as np

sequence = string.ascii_lowercase + "\n" # "abcdefghijklmnopqrstuvwxyz\n"
repeats = 100_000 

data = sequence * repeats

print(f"Data sample (first 100 chars):\n{data[:100]}")
print(f"Total dataset size: {len(data):,} characters")

# --- PREPROCESS ---
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Unique characters: {len(chars)} ({''.join(chars).replace(chr(10), '\\n')})")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

# Split 90/10
n = len(data)
cutoff = int(n * 0.9)
train_data = data[:cutoff]
val_data = data[cutoff:]

train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"Train tokens: {len(train_ids):,}")
print(f"Val tokens: {len(val_ids):,}")

# --- SAVE ---
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save meta
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done.")

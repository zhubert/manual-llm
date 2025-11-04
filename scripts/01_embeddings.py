"""
Calculate token and position embeddings for our input sequence.

Input: "I like transformers"
Tokens: [1, 3, 4, 5, 2]  (<BOS>, I, like, transformers, <EOS>)
"""

import random

# Set seed for reproducibility
random.seed(42)

# Model hyperparameters
VOCAB_SIZE = 6
D_MODEL = 16
MAX_SEQ_LEN = 5

# Input sequence
tokens = [1, 3, 4, 5, 2]  # <BOS>, I, like, transformers, <EOS>
seq_len = len(tokens)

# Helper function to generate random vector
def random_vector(size, scale=0.1):
    """Generate a random vector with values ~ N(0, scale^2)"""
    return [random.gauss(0, scale) for _ in range(size)]

# Helper function to add two vectors
def add_vectors(v1, v2):
    """Element-wise addition of two vectors"""
    return [a + b for a, b in zip(v1, v2)]

# Helper function to format vector for printing
def format_vector(vec):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

print("=" * 80)
print("EMBEDDING CALCULATION")
print("=" * 80)

# Initialize embedding matrices with small random values
print("\n1. Initialize Token Embedding Matrix E_token [6, 16]")
print("-" * 80)
E_token = [random_vector(D_MODEL) for _ in range(VOCAB_SIZE)]
print(f"Shape: [{VOCAB_SIZE}, {D_MODEL}]")
print("\nE_token =")
for i, row in enumerate(E_token):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][i]
    print(f"  Token {i} ({token_name:12s}): {format_vector(row)}")

print("\n" + "=" * 80)
print("2. Initialize Position Embedding Matrix E_pos [5, 16]")
print("-" * 80)
E_pos = [random_vector(D_MODEL) for _ in range(MAX_SEQ_LEN)]
print(f"Shape: [{MAX_SEQ_LEN}, {D_MODEL}]")
print("\nE_pos =")
for i, row in enumerate(E_pos):
    print(f"  Position {i}: {format_vector(row)}")

print("\n" + "=" * 80)
print("3. Look up Token Embeddings for Our Sequence")
print("-" * 80)
print(f"Sequence token IDs: {tokens}")
print()

token_embeddings = []
for i, token_id in enumerate(tokens):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][token_id]
    emb = E_token[token_id]
    token_embeddings.append(emb)
    print(f"Position {i}: Token {token_id} ({token_name:12s})")
    print(f"  E_token[{token_id}] = {format_vector(emb)}")
    print()

print("\n" + "=" * 80)
print("4. Add Position Embeddings to Token Embeddings")
print("-" * 80)
print("X[i] = E_token[token_id[i]] + E_pos[i]\n")

X = [add_vectors(token_embeddings[i], E_pos[i]) for i in range(seq_len)]

for i in range(seq_len):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    print(f"Position {i} ('{token_name}'):")
    print(f"  Token embedding:    {format_vector(token_embeddings[i])}")
    print(f"  Position embedding: {format_vector(E_pos[i])}")
    print(f"  Combined X[{i}]:       {format_vector(X[i])}")
    print()

print("=" * 80)
print("5. FINAL COMBINED EMBEDDINGS MATRIX X")
print("=" * 80)
print(f"Shape: [{seq_len}, {D_MODEL}] (seq_len, d_model)")
print("\nX =")
for i, row in enumerate(X):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    print(f"  {format_vector(row)}  # pos {i}: {token_name}")
print()

# Print in a format easy to copy to docs
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)
print("```")
print("X = [")
for i, row in enumerate(X):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    values = ", ".join([f"{v:7.4f}" for v in row])
    print(f"  [{values}],  # pos {i}: {token_name}")
print("]")
print("```")

# Save for use in next steps using pickle
import pickle
print("\n" + "=" * 80)
print("Saving matrices for next steps...")
print("=" * 80)
data = {
    'X': X,
    'E_token': E_token,
    'E_pos': E_pos,
    'tokens': tokens
}
with open('data/embeddings.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/embeddings.pkl")

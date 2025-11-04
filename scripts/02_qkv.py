"""
Calculate Query, Key, Value projections for multi-head attention.

We have 2 attention heads, each with dimension d_k = d_model / num_heads = 8.
Each head has its own weight matrices W_Q, W_K, W_V of shape [16, 8].
"""

import random
import pickle

# Set seed for reproducibility
random.seed(42)

# Model hyperparameters
D_MODEL = 16
NUM_HEADS = 2
D_K = D_MODEL // NUM_HEADS  # 8

print("=" * 80)
print("QKV PROJECTION CALCULATION")
print("=" * 80)

# Load embeddings from previous step
print("\n1. Load Embeddings Matrix X from Previous Step")
print("-" * 80)
with open('data/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    X = data['X']
    tokens = data['tokens']

seq_len = len(X)
print(f"Shape of X: [{seq_len}, {D_MODEL}] (seq_len, d_model)")
print(f"Number of heads: {NUM_HEADS}")
print(f"Dimension per head (d_k): {D_K}")

# Helper function to generate random matrix
def random_matrix(rows, cols, scale=0.1):
    """Generate a random matrix with values ~ N(0, scale^2)"""
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

# Helper function to format vector for printing
def format_vector(vec):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

# Helper function to format matrix for printing
def format_matrix(mat):
    """Format matrix as string with 4 decimal places"""
    result = "[\n"
    for row in mat:
        result += "  " + format_vector(row) + "\n"
    result += "]"
    return result

# Helper function for matrix-vector multiplication
def matmul_vec(mat, vec):
    """Multiply matrix by vector: result[i] = sum(mat[i][j] * vec[j])"""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

# Helper function for matrix multiplication
def matmul(A, B):
    """Multiply matrices A @ B where A is [m, n] and B is [n, p]"""
    m, n = len(A), len(A[0])
    p = len(B[0])
    result = [[0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return result

# Initialize weight matrices for each head
print("\n" + "=" * 80)
print("2. Initialize Weight Matrices for Each Head")
print("=" * 80)

W_Q = []
W_K = []
W_V = []

for head in range(NUM_HEADS):
    print(f"\n--- Head {head} ---")
    print(f"W_Q[{head}]: [{D_MODEL}, {D_K}]")
    w_q = random_matrix(D_MODEL, D_K)
    W_Q.append(w_q)
    print(f"First 3 rows of W_Q[{head}]:")
    for i in range(min(3, D_MODEL)):
        print(f"  Row {i}: {format_vector(w_q[i])}")

    print(f"\nW_K[{head}]: [{D_MODEL}, {D_K}]")
    w_k = random_matrix(D_MODEL, D_K)
    W_K.append(w_k)
    print(f"First 3 rows of W_K[{head}]:")
    for i in range(min(3, D_MODEL)):
        print(f"  Row {i}: {format_vector(w_k[i])}")

    print(f"\nW_V[{head}]: [{D_MODEL}, {D_K}]")
    w_v = random_matrix(D_MODEL, D_K)
    W_V.append(w_v)
    print(f"First 3 rows of W_V[{head}]:")
    for i in range(min(3, D_MODEL)):
        print(f"  Row {i}: {format_vector(w_v[i])}")

# Compute Q, K, V for each head
print("\n" + "=" * 80)
print("3. Compute Q, K, V Projections for Each Head")
print("=" * 80)

Q_all = []
K_all = []
V_all = []

for head in range(NUM_HEADS):
    print(f"\n{'=' * 80}")
    print(f"HEAD {head}")
    print('=' * 80)

    # Q = X @ W_Q
    print(f"\n3.{head+1}a. Compute Query Matrix Q for Head {head}")
    print("-" * 80)
    print(f"Q[{head}] = X @ W_Q[{head}]")
    print(f"Shape: [{seq_len}, {D_MODEL}] @ [{D_MODEL}, {D_K}] = [{seq_len}, {D_K}]")

    Q = matmul(X, W_Q[head])
    Q_all.append(Q)

    print("\nDetailed calculation for position 0:")
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[0]]
    print(f"Q[{head}][0] (for '{token_name}'):")
    print(f"  X[0] = {format_vector(X[0])}")
    print(f"  Q[{head}][0] = X[0] @ W_Q[{head}]")
    print(f"  Result: {format_vector(Q[0])}")

    print(f"\nComplete Q[{head}] matrix:")
    for i, row in enumerate(Q):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        print(f"  {format_vector(row)}  # pos {i}: {token_name}")

    # K = X @ W_K
    print(f"\n3.{head+1}b. Compute Key Matrix K for Head {head}")
    print("-" * 80)
    print(f"K[{head}] = X @ W_K[{head}]")
    print(f"Shape: [{seq_len}, {D_MODEL}] @ [{D_MODEL}, {D_K}] = [{seq_len}, {D_K}]")

    K = matmul(X, W_K[head])
    K_all.append(K)

    print("\nDetailed calculation for position 0:")
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[0]]
    print(f"K[{head}][0] (for '{token_name}'):")
    print(f"  X[0] = {format_vector(X[0])}")
    print(f"  K[{head}][0] = X[0] @ W_K[{head}]")
    print(f"  Result: {format_vector(K[0])}")

    print(f"\nComplete K[{head}] matrix:")
    for i, row in enumerate(K):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        print(f"  {format_vector(row)}  # pos {i}: {token_name}")

    # V = X @ W_V
    print(f"\n3.{head+1}c. Compute Value Matrix V for Head {head}")
    print("-" * 80)
    print(f"V[{head}] = X @ W_V[{head}]")
    print(f"Shape: [{seq_len}, {D_MODEL}] @ [{D_MODEL}, {D_K}] = [{seq_len}, {D_K}]")

    V = matmul(X, W_V[head])
    V_all.append(V)

    print("\nDetailed calculation for position 0:")
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[0]]
    print(f"V[{head}][0] (for '{token_name}'):")
    print(f"  X[0] = {format_vector(X[0])}")
    print(f"  V[{head}][0] = X[0] @ W_V[{head}]")
    print(f"  Result: {format_vector(V[0])}")

    print(f"\nComplete V[{head}] matrix:")
    for i, row in enumerate(V):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        print(f"  {format_vector(row)}  # pos {i}: {token_name}")

# Print formatted for documentation
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)

for head in range(NUM_HEADS):
    print(f"\n### Head {head}\n")
    print("```python")
    print(f"Q[{head}] = [")
    for i, row in enumerate(Q_all[head]):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        values = ", ".join([f"{v:7.4f}" for v in row])
        print(f"  [{values}],  # pos {i}: {token_name}")
    print("]")
    print()
    print(f"K[{head}] = [")
    for i, row in enumerate(K_all[head]):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        values = ", ".join([f"{v:7.4f}" for v in row])
        print(f"  [{values}],  # pos {i}: {token_name}")
    print("]")
    print()
    print(f"V[{head}] = [")
    for i, row in enumerate(V_all[head]):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        values = ", ".join([f"{v:7.4f}" for v in row])
        print(f"  [{values}],  # pos {i}: {token_name}")
    print("]")
    print("```")

# Save for use in next steps
print("\n" + "=" * 80)
print("Saving matrices for next steps...")
print("=" * 80)
data = {
    'X': X,
    'tokens': tokens,
    'W_Q': W_Q,
    'W_K': W_K,
    'W_V': W_V,
    'Q': Q_all,
    'K': K_all,
    'V': V_all
}
with open('data/qkv.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/qkv.pkl")

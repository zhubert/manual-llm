"""
Calculate attention mechanism for both heads.

Steps:
1. Compute attention scores: Q @ K^T
2. Scale by sqrt(d_k)
3. Apply causal mask (for autoregressive generation)
4. Apply softmax to get attention weights
5. Compute attention output: weights @ V
"""

import pickle
import math

# Model hyperparameters
D_MODEL = 16
NUM_HEADS = 2
D_K = D_MODEL // NUM_HEADS  # 8

print("=" * 80)
print("ATTENTION MECHANISM CALCULATION")
print("=" * 80)

# Load Q, K, V from previous step
print("\n1. Load Q, K, V from Previous Step")
print("-" * 80)
with open('data/qkv.pkl', 'rb') as f:
    data = pickle.load(f)
    Q = data['Q']
    K = data['K']
    V = data['V']
    tokens = data['tokens']
    X = data['X']

seq_len = len(Q[0])
print(f"Sequence length: {seq_len}")
print(f"Number of heads: {NUM_HEADS}")
print(f"Dimension per head (d_k): {D_K}")

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

# Helper function for matrix transpose
def transpose(mat):
    """Transpose a matrix"""
    return [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]

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

# Helper function for element-wise matrix operation
def add_scalar_to_matrix(mat, scalar):
    """Add scalar to all elements of matrix"""
    return [[mat[i][j] + scalar for j in range(len(mat[0]))] for i in range(len(mat))]

# Helper function for softmax
def softmax(vec):
    """Compute softmax of a vector"""
    # Subtract max for numerical stability
    max_val = max(vec)
    exp_vec = [math.exp(v - max_val) for v in vec]
    sum_exp = sum(exp_vec)
    return [e / sum_exp for e in exp_vec]

# Compute attention for each head
attention_outputs = []
attention_weights_all = []

for head in range(NUM_HEADS):
    print(f"\n{'=' * 80}")
    print(f"HEAD {head}")
    print('=' * 80)

    # Step 1: Compute raw attention scores Q @ K^T
    print(f"\n2.{head+1}a. Compute Attention Scores: Q @ K^T")
    print("-" * 80)
    print(f"Scores[{head}] = Q[{head}] @ K[{head}]^T")
    print(f"Shape: [{seq_len}, {D_K}] @ [{D_K}, {seq_len}] = [{seq_len}, {seq_len}]")

    K_T = transpose(K[head])
    scores_raw = matmul(Q[head], K_T)

    print("\nDetailed calculation for score[0, 0] (BOS attending to BOS):")
    print(f"  Q[{head}][0] = {format_vector(Q[head][0])}")
    print(f"  K[{head}][0] = {format_vector(K[head][0])}")
    print(f"  score[0, 0] = dot(Q[{head}][0], K[{head}][0])")
    dot_product = sum(Q[head][0][k] * K[head][0][k] for k in range(D_K))
    print(f"  score[0, 0] = {dot_product:.4f}")

    print(f"\nRaw attention scores matrix (before scaling):")
    for i, row in enumerate(scores_raw):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        print(f"  {format_vector(row)}  # pos {i}: {token_name}")

    # Step 2: Scale by sqrt(d_k)
    print(f"\n2.{head+1}b. Scale by sqrt(d_k) = sqrt({D_K}) = {math.sqrt(D_K):.4f}")
    print("-" * 80)
    scale_factor = math.sqrt(D_K)
    scores_scaled = [[scores_raw[i][j] / scale_factor for j in range(seq_len)]
                     for i in range(seq_len)]

    print(f"Scaled scores[0, 0] = {scores_raw[0][0]:.4f} / {scale_factor:.4f} = {scores_scaled[0][0]:.4f}")
    print(f"\nScaled attention scores:")
    for i, row in enumerate(scores_scaled):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        print(f"  {format_vector(row)}  # pos {i}: {token_name}")

    # Step 3: Apply causal mask
    print(f"\n2.{head+1}c. Apply Causal Mask")
    print("-" * 80)
    print("For autoregressive generation, each position can only attend to")
    print("previous positions (including itself). We set future positions to -inf.")
    print("\nMask pattern (True = allowed, False = masked):")
    mask = [[j <= i for j in range(seq_len)] for i in range(seq_len)]
    for i, row in enumerate(mask):
        mask_str = "".join(["✓" if m else "✗" for m in row])
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        print(f"  {mask_str}  # pos {i}: {token_name}")

    # Apply mask by setting masked positions to very negative number
    MASK_VALUE = -1e9
    scores_masked = [[scores_scaled[i][j] if j <= i else MASK_VALUE
                      for j in range(seq_len)] for i in range(seq_len)]

    print(f"\nMasked scores (future positions set to {MASK_VALUE}):")
    for i, row in enumerate(scores_masked):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        formatted = "[" + ", ".join([f"{v:7.4f}" if v > -1e8 else "    -inf" for v in row]) + "]"
        print(f"  {formatted}  # pos {i}: {token_name}")

    # Step 4: Apply softmax
    print(f"\n2.{head+1}d. Apply Softmax")
    print("-" * 80)
    print("Convert scores to probabilities (each row sums to 1)")

    attention_weights = [softmax(scores_masked[i]) for i in range(seq_len)]

    print("\nExample: softmax for position 1 (token 'I'):")
    print(f"  Input scores:  {format_vector(scores_masked[1][:2])} (only first 2 shown)")
    print(f"  After softmax: {format_vector(attention_weights[1][:2])}")
    print(f"  Sum of row: {sum(attention_weights[1]):.6f} (should be 1.0)")

    print(f"\nAttention weights (each row sums to 1):")
    for i, row in enumerate(attention_weights):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        print(f"  {format_vector(row)}  # pos {i}: {token_name}")

    attention_weights_all.append(attention_weights)

    # Step 5: Compute attention output
    print(f"\n2.{head+1}e. Compute Attention Output: weights @ V")
    print("-" * 80)
    print(f"Output[{head}] = attention_weights @ V[{head}]")
    print(f"Shape: [{seq_len}, {seq_len}] @ [{seq_len}, {D_K}] = [{seq_len}, {D_K}]")

    attention_output = matmul(attention_weights, V[head])
    attention_outputs.append(attention_output)

    print("\nDetailed calculation for output[0] (BOS position):")
    print(f"  weights[0] = {format_vector(attention_weights[0])}")
    print(f"  output[0] = weighted sum of all V vectors:")
    for j in range(min(2, seq_len)):
        print(f"    {attention_weights[0][j]:.4f} * V[{head}][{j}] = {format_vector([attention_weights[0][j] * V[head][j][k] for k in range(D_K)])}")
    print(f"  Result: {format_vector(attention_output[0])}")

    print(f"\nComplete attention output for head {head}:")
    for i, row in enumerate(attention_output):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        print(f"  {format_vector(row)}  # pos {i}: {token_name}")

# Print formatted for documentation
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)

for head in range(NUM_HEADS):
    print(f"\n### Head {head}\n")

    # Recompute for documentation
    K_T = transpose(K[head])
    scores_raw = matmul(Q[head], K_T)
    scale_factor = math.sqrt(D_K)
    scores_scaled = [[scores_raw[i][j] / scale_factor for j in range(seq_len)]
                     for i in range(seq_len)]
    MASK_VALUE = -1e9
    scores_masked = [[scores_scaled[i][j] if j <= i else MASK_VALUE
                      for j in range(seq_len)] for i in range(seq_len)]
    attention_weights = [softmax(scores_masked[i]) for i in range(seq_len)]

    print("#### Attention Weights\n")
    print("```python")
    print(f"attention_weights[{head}] = [")
    for i, row in enumerate(attention_weights):
        token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
        values = ", ".join([f"{v:7.4f}" for v in row])
        print(f"  [{values}],  # pos {i}: {token_name}")
    print("]")
    print("```")

    print("\n#### Attention Output\n")
    print("```python")
    print(f"attention_output[{head}] = [")
    for i, row in enumerate(attention_outputs[head]):
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
    'Q': Q,
    'K': K,
    'V': V,
    'attention_weights': attention_weights_all,
    'attention_output': attention_outputs
}
with open('data/attention.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/attention.pkl")

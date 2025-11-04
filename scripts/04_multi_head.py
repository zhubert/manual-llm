"""
Calculate multi-head attention output.

Steps:
1. Concatenate outputs from all attention heads
2. Apply output projection W_o to project back to d_model dimensions
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
print("MULTI-HEAD ATTENTION CALCULATION")
print("=" * 80)

# Load attention outputs from previous step
print("\n1. Load Attention Outputs from Previous Step")
print("-" * 80)
with open('data/attention.pkl', 'rb') as f:
    data = pickle.load(f)
    attention_outputs = data['attention_output']
    tokens = data['tokens']
    X = data['X']

seq_len = len(attention_outputs[0])
print(f"Sequence length: {seq_len}")
print(f"Number of heads: {NUM_HEADS}")
print(f"Dimension per head (d_k): {D_K}")
print(f"Shape of each head output: [{seq_len}, {D_K}]")

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

# Step 2: Concatenate attention outputs from all heads
print("\n" + "=" * 80)
print("2. Concatenate Head Outputs")
print("=" * 80)
print(f"Concatenate along feature dimension: [{seq_len}, {D_K}] + [{seq_len}, {D_K}] -> [{seq_len}, {D_MODEL}]")

# Concatenate head outputs for each position
concat_output = []
for pos in range(seq_len):
    # For this position, concatenate vectors from all heads
    concat_vec = []
    for head in range(NUM_HEADS):
        concat_vec.extend(attention_outputs[head][pos])
    concat_output.append(concat_vec)

print(f"\nShape after concatenation: [{seq_len}, {D_MODEL}]")

print("\nExample: Position 0 (BOS)")
print(f"  Head 0 output: {format_vector(attention_outputs[0][0])}")
print(f"  Head 1 output: {format_vector(attention_outputs[1][0])}")
print(f"  Concatenated:  {format_vector(concat_output[0])}")

print("\nAll concatenated outputs:")
for i, vec in enumerate(concat_output):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    print(f"  {format_vector(vec)}  # pos {i}: {token_name}")

# Step 3: Initialize output projection matrix W_o
print("\n" + "=" * 80)
print("3. Initialize Output Projection Matrix W_o")
print("=" * 80)
print(f"Shape: [{D_MODEL}, {D_MODEL}]")
print("Projects concatenated heads back to d_model dimensions")

W_o = random_matrix(D_MODEL, D_MODEL)

print("\nW_o = ")
for i, row in enumerate(W_o):
    print(f"  {format_vector(row)}")

# Step 4: Apply output projection
print("\n" + "=" * 80)
print("4. Apply Output Projection")
print("=" * 80)
print(f"multi_head_output = concat_output @ W_o^T")
print(f"Shape: [{seq_len}, {D_MODEL}] @ [{D_MODEL}, {D_MODEL}] = [{seq_len}, {D_MODEL}]")

# Transpose W_o for matrix multiplication
W_o_T = [[W_o[j][i] for j in range(D_MODEL)] for i in range(D_MODEL)]

# Apply projection
multi_head_output = matmul(concat_output, W_o_T)

print("\nExample calculation for position 0 (BOS):")
print(f"  concat_output[0] = {format_vector(concat_output[0])}")
print(f"  Multiply by W_o^T (showing first few elements)...")
print(f"  Result[0] = {format_vector(multi_head_output[0])}")

print("\n" + "=" * 80)
print("MULTI-HEAD ATTENTION OUTPUT")
print("=" * 80)
print(f"Shape: [{seq_len}, {D_MODEL}]")
print("\nThis is the final output after combining both attention heads:")
for i, vec in enumerate(multi_head_output):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    print(f"  {format_vector(vec)}  # pos {i}: {token_name}")

# Print formatted for documentation
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)

print("\n### Concatenated Head Outputs\n")
print("```python")
print("concat_output = [")
for i, vec in enumerate(concat_output):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    values = ", ".join([f"{v:7.4f}" for v in vec])
    print(f"  [{values}],  # pos {i}: {token_name}")
print("]")
print("```")

print("\n### Multi-Head Attention Output\n")
print("```python")
print("multi_head_output = [")
for i, vec in enumerate(multi_head_output):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    values = ", ".join([f"{v:7.4f}" for v in vec])
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
    'attention_outputs': attention_outputs,
    'concat_output': concat_output,
    'W_o': W_o,
    'multi_head_output': multi_head_output
}
with open('data/multi_head.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/multi_head.pkl")

"""
Calculate gradients through the language modeling head (output projection).

We have gradients dL/dlogits from the previous step. Now we need to:
1. Compute gradients for W_lm (the LM head weights)
2. Backpropagate to the hidden states (layer norm output)

Forward pass: logits = W_lm @ hidden_states
  Where W_lm is [vocab_size, d_model] = [6, 16]

Backward pass (chain rule):
  dL/dW_lm = dL/dlogits @ hidden_states^T  (outer product for each position)
  dL/dhidden_states = W_lm^T @ dL/dlogits  (backprop through linear layer)
"""

import pickle

print("=" * 80)
print("BACKWARD PASS: OUTPUT LAYER (LM HEAD)")
print("=" * 80)

# Load previous gradients and forward pass data
print("\n1. Load Data from Previous Steps")
print("-" * 80)
with open('data/grad_loss.pkl', 'rb') as f:
    data = pickle.load(f)
    dL_dlogits = data['dL_dlogits']
    tokens = data['tokens']
    targets = data['targets']
    X = data['X']

with open('data/layernorm.pkl', 'rb') as f:
    data = pickle.load(f)
    layer_norm_output = data['layer_norm_output']

with open('data/loss.pkl', 'rb') as f:
    data = pickle.load(f)
    W_lm = data['W_lm']

vocab_size, d_model = len(W_lm), len(W_lm[0])
seq_len = len(tokens)

print(f"Sequence length: {seq_len}")
print(f"Vocab size: {vocab_size}")
print(f"d_model: {d_model}")
print(f"W_lm shape: [{vocab_size}, {d_model}]")

vocab = ['<PAD>', '<BOS>', '<EOS>', 'I', 'like', 'transformers']

# Helper functions
def format_vector(vec):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

def outer_product(v1, v2):
    """Compute outer product: v1 @ v2^T"""
    return [[v1[i] * v2[j] for j in range(len(v2))] for i in range(len(v1))]

def matmul_vec(mat, vec):
    """Matrix-vector multiplication"""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

def matmul_T_vec(mat, vec):
    """Multiply by transpose of matrix: mat^T @ vec"""
    return [sum(mat[j][i] * vec[j] for j in range(len(mat))) for i in range(len(mat[0]))]

def add_matrices(m1, m2):
    """Element-wise matrix addition"""
    return [[m1[i][j] + m2[i][j] for j in range(len(m1[0]))] for i in range(len(m1))]

# Step 2: Compute gradients for W_lm
print("\n" + "=" * 80)
print("2. Compute Gradients for W_lm (LM Head Weights)")
print("=" * 80)
print("Forward pass: logits[pos] = W_lm @ hidden_states[pos]")
print()
print("By the chain rule:")
print("  dL/dW_lm += dL/dlogits[pos] @ hidden_states[pos]^T")
print()
print("We sum gradients across all positions (batched gradient accumulation).")

# Initialize gradient matrix
dL_dW_lm = [[0.0 for _ in range(d_model)] for _ in range(vocab_size)]

for pos in range(len(targets)):  # Only positions with targets
    # Outer product: dL_dlogits[pos] @ layer_norm_output[pos]^T
    grad_contribution = outer_product(dL_dlogits[pos], layer_norm_output[pos])

    # Accumulate
    dL_dW_lm = add_matrices(dL_dW_lm, grad_contribution)

    if pos == 0:  # Show detailed calculation
        current_token = vocab[tokens[pos]]
        target_token = vocab[targets[pos]]
        print(f"\n--- Position {pos} ('{current_token}' -> '{target_token}') ---")
        print(f"  dL/dlogits[{pos}]: {format_vector(dL_dlogits[pos])}")
        print(f"  hidden_states[{pos}] (first 8): {format_vector(layer_norm_output[pos][:8])}...")
        print()
        print(f"  Computing outer product [6, 1] @ [1, 16] = [6, 16]")
        print(f"  Contribution to dL/dW_lm (row 0, first 8 cols):")
        print(f"    {format_vector(grad_contribution[0][:8])}...")

print(f"\n  Accumulated dL/dW_lm (row 0, first 8 cols): {format_vector(dL_dW_lm[0][:8])}...")
print(f"  Shape: [{vocab_size}, {d_model}]")

# Step 3: Compute gradients for hidden states
print("\n" + "=" * 80)
print("3. Backpropagate to Hidden States (Layer Norm Output)")
print("=" * 80)
print("Forward pass: logits = W_lm @ hidden_states")
print()
print("By the chain rule:")
print("  dL/dhidden_states = W_lm^T @ dL/dlogits")
print()
print("This backpropagates the gradient through the linear layer.")

dL_dlayer_norm_output = []

for pos in range(len(targets)):
    # Multiply: W_lm^T @ dL_dlogits[pos]
    grad = matmul_T_vec(W_lm, dL_dlogits[pos])
    dL_dlayer_norm_output.append(grad)

    if pos == 0:  # Show detailed calculation
        current_token = vocab[tokens[pos]]
        target_token = vocab[targets[pos]]
        print(f"\n--- Position {pos} ('{current_token}' -> '{target_token}') ---")
        print(f"  dL/dlogits[{pos}]: {format_vector(dL_dlogits[pos])}")
        print()
        print(f"  Computing W_lm^T @ dL/dlogits")
        print(f"  Shape: [16, 6] @ [6] = [16]")
        print()
        print(f"  dL/dhidden_states[{pos}]: {format_vector(grad)}")

print("\n" + "=" * 80)
print("ALL HIDDEN STATE GRADIENTS")
print("=" * 80)
for i, grad in enumerate(dL_dlayer_norm_output):
    current_token = vocab[tokens[i]]
    target_token = vocab[targets[i]]
    print(f"  {format_vector(grad)}  # pos {i}: {current_token} -> {target_token}")

# Print formatted for documentation
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)

print("\n### dL/dW_lm (first 3 rows, first 8 columns):\n")
print("```python")
for i in range(min(3, vocab_size)):
    values = ", ".join([f"{dL_dW_lm[i][j]:7.4f}" for j in range(min(8, d_model))])
    print(f"  [{values}, ...]  # row {i} ({vocab[i]})")
print("```")

print("\n### dL/dlayer_norm_output:\n")
print("```python")
print("dL_dlayer_norm_output = [")
for i, grad in enumerate(dL_dlayer_norm_output):
    current_token = vocab[tokens[i]]
    target_token = vocab[targets[i]]
    values = ", ".join([f"{v:7.4f}" for v in grad])
    print(f"  [{values}],  # pos {i}: {current_token} -> {target_token}")
print("]")
print("```")

# Save for next step
print("\n" + "=" * 80)
print("Saving gradients for next step...")
print("=" * 80)

# We need to extend dL_dlayer_norm_output to all positions (including position 4 which has no target)
# For position 4 (<EOS>), there's no loss contribution, so gradient is zero
dL_dlayer_norm_output_full = dL_dlayer_norm_output + [[0.0] * d_model]

data = {
    'X': X,
    'tokens': tokens,
    'targets': targets,
    'dL_dW_lm': dL_dW_lm,
    'dL_dlayer_norm_output': dL_dlayer_norm_output_full,
    'W_lm': W_lm
}
with open('data/grad_output.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/grad_output.pkl")
print("\nNext step: backpropagate through layer normalization and residual connections")

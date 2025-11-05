"""
Calculate gradients through layer normalization, residual connections, and FFN.

This is a complex backward pass step that handles:
1. Layer normalization (with gamma, beta parameters)
2. Residual connection (split gradient into two paths)
3. Feed-forward network (W2, b2, GELU, W1, b1)

We'll work backwards from dL/dlayer_norm_output to dL/dmulti_head_output.
"""

import pickle
import math

print("=" * 80)
print("BACKWARD PASS: LAYER NORM, RESIDUAL, AND FFN")
print("=" * 80)

# Load data
print("\n1. Load Data from Previous Steps")
print("-" * 80)
with open('data/grad_output.pkl', 'rb') as f:
    data = pickle.load(f)
    dL_dlayer_norm_output = data['dL_dlayer_norm_output']
    tokens = data['tokens']
    X = data['X']

with open('data/layernorm.pkl', 'rb') as f:
    data = pickle.load(f)
    layer_norm_output = data['layer_norm_output']
    residual_output = data['residual_output']
    gamma = data['gamma']
    beta = data['beta']
    multi_head_output = data['multi_head_output']
    ffn_output = data['ffn_output']

with open('data/feedforward.pkl', 'rb') as f:
    data = pickle.load(f)
    W1 = data['W1']
    b1 = data['b1']
    W2 = data['W2']
    b2 = data['b2']

d_model = 16
d_ff = 64
seq_len = len(tokens)

print(f"Sequence length: {seq_len}")
print(f"d_model: {d_model}, d_ff: {d_ff}")

vocab = ['<PAD>', '<BOS>', '<EOS>', 'I', 'like', 'transformers']

# Helper functions
def format_vector(vec, n=16):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec[:n]]) + ("]" if n == len(vec) else ", ...]")

def gelu_derivative(x):
    """Derivative of GELU activation"""
    # GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    # This derivative is complex, so we'll use a numerical approximation
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    x_cubed = x * x * x
    tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
    tanh_val = math.tanh(tanh_arg)

    # Derivative involves both the tanh and its derivative
    sech_squared = 1.0 - tanh_val * tanh_val  # sech²(x) = 1 - tanh²(x)
    dtanh_dx = sqrt_2_over_pi * (1.0 + 0.044715 * 3.0 * x * x)

    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech_squared * dtanh_dx

print("\n" + "=" * 80)
print("2. Backward Through Layer Normalization")
print("=" * 80)
print("Layer norm is complex because each output depends on ALL inputs (due to mean/variance).")
print("Similar to softmax, we need the full Jacobian!")
print()
print("Layer Norm formula:")
print("  1. μ = mean(x)")
print("  2. σ² = variance(x)")
print("  3. x_norm = (x - μ) / √(σ² + ε)")
print("  4. y = γ ⊙ x_norm + β")
print()
print("The backward pass requires computing:")
print("  dL/dx = (γ/σ) ⊙ (dL/dy - mean(dL/dy) - x_norm ⊙ mean(x_norm ⊙ dL/dy))")
print()
print("Where the mean operations account for the Jacobian of the normalization.")

# Compute gradients for gamma and beta
dL_dgamma = [0.0] * d_model
dL_dbeta = [0.0] * d_model

# Need to recompute normalized values from forward pass
normalized = []
means = []
stds = []

for pos in range(seq_len):
    mean_val = sum(residual_output[pos]) / d_model
    var_val = sum((x - mean_val) ** 2 for x in residual_output[pos]) / d_model
    std_val = math.sqrt(var_val + 1e-5)

    means.append(mean_val)
    stds.append(std_val)

    x_norm = [(x - mean_val) / std_val for x in residual_output[pos]]
    normalized.append(x_norm)

    # Accumulate gamma and beta gradients
    for i in range(d_model):
        dL_dgamma[i] += dL_dlayer_norm_output[pos][i] * x_norm[i]
        dL_dbeta[i] += dL_dlayer_norm_output[pos][i]

print(f"  dL/dgamma: {format_vector(dL_dgamma)}")
print(f"  dL/dbeta: {format_vector(dL_dbeta)}")

# Backprop through layer norm using full Jacobian
dL_dresidual = []

for pos in range(seq_len):
    # Gradient w.r.t. normalized values
    dL_dx_norm = [gamma[i] * dL_dlayer_norm_output[pos][i] for i in range(d_model)]

    # Compute terms for Jacobian
    mean_dL_dx_norm = sum(dL_dx_norm) / d_model
    mean_xnorm_dL_dx_norm = sum(normalized[pos][i] * dL_dx_norm[i] for i in range(d_model)) / d_model

    # Apply layer norm Jacobian
    dL_dx = []
    for i in range(d_model):
        grad = (1.0 / stds[pos]) * (
            dL_dx_norm[i] - mean_dL_dx_norm - normalized[pos][i] * mean_xnorm_dL_dx_norm
        )
        dL_dx.append(grad)

    dL_dresidual.append(dL_dx)

print(f"\n  Example gradient computation for position 0:")
print(f"    mean = {means[0]:.6f}, std = {stds[0]:.6f}")
print(f"    mean(dL/dx_norm) = {sum(dL_dx_norm) / d_model:.6f}")
print(f"    dL/dresidual_output[0]: {format_vector(dL_dresidual[0])}")

print("\n" + "=" * 80)
print("3. Backward Through Residual Connection")
print("=" * 80)
print("Forward: residual = multi_head_output + ffn_output")
print("Backward: gradient splits into two paths (addition distributes gradients)")

dL_dmulti_head = dL_dresidual  # Gradient flows to attention path
dL_dffn_output = dL_dresidual  # Gradient flows to FFN path

print(f"  dL/dmulti_head_output[0]: {format_vector(dL_dmulti_head[0])}")
print(f"  dL/dffn_output[0]: {format_vector(dL_dffn_output[0])}")

# Now backprop through FFN for each position
print("\n" + "=" * 80)
print("4. Backward Through FFN (Feed-Forward Network)")
print("=" * 80)
print("FFN has three stages:")
print("  1. Linear layer 2: output = W2 @ activated + b2")
print("  2. GELU activation: activated = GELU(hidden)")
print("  3. Linear layer 1: hidden = W1 @ input + b1")

# We need to recompute forward pass intermediate values for backprop
print("\nRecomputing FFN forward pass (to get intermediate activations)...")

# Helper for forward pass
def matmul_vec(mat, vec):
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

def add_vectors(v1, v2):
    return [a + b for a, b in zip(v1, v2)]

def gelu(x):
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))

# Recompute activations
ffn_hidden = []
ffn_activated = []

for pos in range(seq_len):
    hidden = matmul_vec(W1, multi_head_output[pos])
    hidden = add_vectors(hidden, b1)
    ffn_hidden.append(hidden)

    activated = [gelu(h) for h in hidden]
    ffn_activated.append(activated)

# Step 4a: Backprop through W2 and b2
print("\nStep 4a: Gradient for W2, b2, and backprop to activated")

dL_dW2 = [[0.0] * d_ff for _ in range(d_model)]
dL_db2 = [0.0] * d_model
dL_dactivated = []

for pos in range(seq_len):
    # dL/dW2 += dL/dffn_output[pos] outer ffn_activated[pos]^T
    for i in range(d_model):
        for j in range(d_ff):
            dL_dW2[i][j] += dL_dffn_output[pos][i] * ffn_activated[pos][j]

    # dL/db2 += dL/dffn_output[pos]
    for i in range(d_model):
        dL_db2[i] += dL_dffn_output[pos][i]

    # dL/dactivated = W2^T @ dL/dffn_output
    grad_act = [sum(W2[i][j] * dL_dffn_output[pos][i] for i in range(d_model)) for j in range(d_ff)]
    dL_dactivated.append(grad_act)

print(f"  dL/dW2[0] (first 8): {format_vector(dL_dW2[0], 8)}")
print(f"  dL/db2: {format_vector(dL_db2)}")
print(f"  dL/dactivated[0] (first 8): {format_vector(dL_dactivated[0], 8)}")

# Step 4b: Backprop through GELU
print("\nStep 4b: Backprop through GELU activation")

dL_dhidden = []
for pos in range(seq_len):
    grad_hidden = [dL_dactivated[pos][j] * gelu_derivative(ffn_hidden[pos][j]) for j in range(d_ff)]
    dL_dhidden.append(grad_hidden)

print(f"  dL/dhidden[0] (first 8): {format_vector(dL_dhidden[0], 8)}")

# Step 4c: Backprop through W1 and b1
print("\nStep 4c: Gradient for W1, b1, and backprop to attention output")

dL_dW1 = [[0.0] * d_model for _ in range(d_ff)]
dL_db1 = [0.0] * d_ff
dL_dmulti_head_ffn = []

for pos in range(seq_len):
    # dL/dW1 += dL/dhidden[pos] outer multi_head_output[pos]^T
    for i in range(d_ff):
        for j in range(d_model):
            dL_dW1[i][j] += dL_dhidden[pos][i] * multi_head_output[pos][j]

    # dL/db1 += dL/dhidden[pos]
    for i in range(d_ff):
        dL_db1[i] += dL_dhidden[pos][i]

    # dL/dmulti_head_output (from FFN path) = W1^T @ dL/dhidden
    grad_mh = [sum(W1[i][j] * dL_dhidden[pos][i] for i in range(d_ff)) for j in range(d_model)]
    dL_dmulti_head_ffn.append(grad_mh)

print(f"  dL/dW1[0]: {format_vector(dL_dW1[0])}")
print(f"  dL/db1 (first 8): {format_vector(dL_db1, 8)}")
print(f"  dL/dmulti_head_output (from FFN): {format_vector(dL_dmulti_head_ffn[0])}")

# Step 5: Combine gradients from both paths
print("\n" + "=" * 80)
print("5. Combine Gradients from Residual Paths")
print("=" * 80)
print("Gradient flows through TWO paths:")
print("  1. Direct residual path (from step 3)")
print("  2. Through FFN (from step 4c)")
print("\nWe sum these gradients:")

dL_dmulti_head_total = []
for pos in range(seq_len):
    combined = [dL_dmulti_head[pos][i] + dL_dmulti_head_ffn[pos][i] for i in range(d_model)]
    dL_dmulti_head_total.append(combined)

print(f"  dL/dmulti_head_output[0] (total): {format_vector(dL_dmulti_head_total[0])}")

# Save gradients
print("\n" + "=" * 80)
print("Saving gradients...")
print("=" * 80)

data = {
    'X': X,
    'tokens': tokens,
    'dL_dW1': dL_dW1,
    'dL_db1': dL_db1,
    'dL_dW2': dL_dW2,
    'dL_db2': dL_db2,
    'dL_dgamma': dL_dgamma,
    'dL_dbeta': dL_dbeta,
    'dL_dmulti_head_output': dL_dmulti_head_total
}
with open('data/grad_ffn.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Saved to data/grad_ffn.pkl")
print("\nNext: Backpropagate through multi-head attention (the most complex part!)")

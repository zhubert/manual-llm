"""
Complete backward pass through attention and embeddings.

This script handles the final backward pass steps:
1. Backprop through multi-head attention output projection
2. Backprop through attention mechanism (including softmax Jacobian)
3. Backprop through QKV projections
4. Backprop to embeddings

This implements the COMPLETE, ACCURATE gradients including the softmax Jacobian.
"""

import pickle
import math

print("=" * 80)
print("BACKWARD PASS: ATTENTION AND EMBEDDINGS (COMPLETE)")
print("=" * 80)

# Load data
print("\n1. Load Data from Previous Steps")
print("-" * 80)
with open('data/grad_ffn.pkl', 'rb') as f:
    data = pickle.load(f)
    dL_dmulti_head_output = data['dL_dmulti_head_output']
    tokens = data['tokens']
    X = data['X']

with open('data/embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
    E_token = emb_data['E_token']
    E_pos = emb_data['E_pos']

with open('data/multi_head.pkl', 'rb') as f:
    attn_data = pickle.load(f)
    Wo = attn_data['W_o']
    concat_output = attn_data['concat_output']
    attention_outputs = attn_data['attention_outputs']

with open('data/attention.pkl', 'rb') as f:
    attn_data = pickle.load(f)
    Q = attn_data['Q']
    K = attn_data['K']
    V = attn_data['V']
    attention_weights = attn_data['attention_weights']

with open('data/qkv.pkl', 'rb') as f:
    qkv_data = pickle.load(f)
    Wq = qkv_data['W_Q']
    Wk = qkv_data['W_K']
    Wv = qkv_data['W_V']

d_model = 16
d_head = 8
num_heads = 2
vocab_size = 6
seq_len = len(tokens)
scale_factor = math.sqrt(d_head)

print(f"Sequence length: {seq_len}")
print(f"d_model: {d_model}, num_heads: {num_heads}, d_head: {d_head}")

vocab = ['<PAD>', '<BOS>', '<EOS>', 'I', 'like', 'transformers']

# Helper functions
def format_vector(vec):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

def dot_product(a, b):
    """Compute dot product of two vectors"""
    return sum(x * y for x, y in zip(a, b))

def matmul_vec(mat, vec):
    """mat @ vec"""
    return [dot_product(row, vec) for row in mat]

def matmul_T_vec(mat, vec):
    """mat^T @ vec"""
    return [sum(mat[j][i] * vec[j] for j in range(len(mat))) for i in range(len(mat[0]))]

print("\n" + "=" * 80)
print("2. Backward Through Output Projection W_o")
print("=" * 80)
print("Forward: multi_head_output = concat @ W_o^T")
print("Backward: dL/dconcat = dL/d(multi_head_output) @ W_o")
print("          dL/dW_o = Σ dL/d(multi_head_output)^T ⊗ concat")

# Backprop through output projection
dL_dconcat = []
for pos in range(seq_len):
    grad_concat = matmul_vec(Wo, dL_dmulti_head_output[pos])
    dL_dconcat.append(grad_concat)

print(f"\nExample: dL/dconcat[0] = {format_vector(dL_dconcat[0])}")

# Compute gradient for W_o using outer products
dL_dWo = [[0.0] * (d_head * num_heads) for _ in range(d_model)]
for pos in range(seq_len):
    for i in range(d_model):
        for j in range(d_head * num_heads):
            dL_dWo[i][j] += dL_dmulti_head_output[pos][i] * concat_output[pos][j]

print(f"dL/dWo computed via outer products")
print(f"  dL/dWo[0][0:4] = {format_vector(dL_dWo[0][:4])}")

print("\n" + "=" * 80)
print("3. Split Gradient by Heads")
print("=" * 80)
print("Concat was formed by concatenating head outputs:")
print(f"  concat[pos] = [head0_output[pos], head1_output[pos]]")
print(f"Split dL/dconcat into per-head gradients")

# Split concatenated gradient back to per-head gradients
dL_d_attention_output = []
for head in range(num_heads):
    head_grads = []
    for pos in range(seq_len):
        start_idx = head * d_head
        end_idx = (head + 1) * d_head
        head_grad = dL_dconcat[pos][start_idx:end_idx]
        head_grads.append(head_grad)
    dL_d_attention_output.append(head_grads)

print(f"\nExample: dL/d(attention_output[0])[0] = {format_vector(dL_d_attention_output[0][0])}")

print("\n" + "=" * 80)
print("4. Backward Through Attention (Per Head)")
print("=" * 80)
print("This is where we compute the FULL gradients including softmax Jacobian!")
print()
print("For each head, we backprop through:")
print("  attention_output = attention_weights @ V")
print("Then through softmax (requires Jacobian):")
print("  attention_weights = softmax(scores_masked)")

# Initialize gradient accumulators
dL_dQ = [[[0.0] * d_head for _ in range(seq_len)] for _ in range(num_heads)]
dL_dK = [[[0.0] * d_head for _ in range(seq_len)] for _ in range(num_heads)]
dL_dV = [[[0.0] * d_head for _ in range(seq_len)] for _ in range(num_heads)]

for head in range(num_heads):
    print(f"\n{'='*80}")
    print(f"HEAD {head}")
    print(f"{'='*80}")

    # Step 4a: Backprop through attention_output = attention_weights @ V
    print(f"\nStep 4a: Backprop through attention_output = attention_weights @ V")
    print("-" * 80)

    # Gradient w.r.t. V
    # dL/dV[j] = Σ_i attention_weights[i,j] * dL/d(attention_output)[i]
    for j in range(seq_len):
        for i in range(seq_len):
            for k in range(d_head):
                dL_dV[head][j][k] += attention_weights[head][i][j] * dL_d_attention_output[head][i][k]

    # Gradient w.r.t. attention_weights
    # dL/d(attention_weights)[i,j] = dL/d(attention_output)[i] · V[j]
    dL_d_attention_weights = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            dL_d_attention_weights[i][j] = dot_product(dL_d_attention_output[head][i], V[head][j])

    print(f"  dL/dV[{head}][0] = {format_vector(dL_dV[head][0])}")
    print(f"  dL/d(attention_weights)[{head}][0] = {format_vector(dL_d_attention_weights[0])}")

    # Step 4b: Backprop through softmax - THE JACOBIAN!
    print(f"\nStep 4b: Backprop through softmax(scores_masked)")
    print("-" * 80)
    print("Softmax Jacobian: ∂s_i/∂x_j = s_i * (δ_ij - s_j)")
    print("Using chain rule: dL/dx_j = s_j * (dL/ds_j - Σ_i s_i * dL/ds_i)")

    dL_d_scores_masked = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        # Compute the dot product term: s · dL/ds
        s_dot_dLds = dot_product(attention_weights[head][i], dL_d_attention_weights[i])

        # Apply softmax backward formula for each element
        for j in range(seq_len):
            dL_d_scores_masked[i][j] = attention_weights[head][i][j] * (
                dL_d_attention_weights[i][j] - s_dot_dLds
            )

    print(f"  dL/d(scores_masked)[{head}][0] = {format_vector(dL_d_scores_masked[0])}")

    # Step 4c: Backprop through causal mask
    print(f"\nStep 4c: Backprop through causal mask")
    print("-" * 80)
    print("Gradient only flows through unmasked positions (j <= i)")

    dL_d_scores_scaled = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:  # Only unmasked positions
                dL_d_scores_scaled[i][j] = dL_d_scores_masked[i][j]

    print(f"  dL/d(scores_scaled)[{head}][0] = {format_vector(dL_d_scores_scaled[0])}")

    # Step 4d: Backprop through scaling by sqrt(d_k)
    print(f"\nStep 4d: Backprop through scaling by sqrt(d_k) = {scale_factor:.4f}")
    print("-" * 80)

    dL_d_scores_raw = [[0.0] * seq_len for _ in range(seq_len)]
    for i in range(seq_len):
        for j in range(seq_len):
            dL_d_scores_raw[i][j] = dL_d_scores_scaled[i][j] / scale_factor

    print(f"  dL/d(scores_raw)[{head}][0] = {format_vector(dL_d_scores_raw[0])}")

    # Step 4e: Backprop through Q @ K^T
    print(f"\nStep 4e: Backprop through scores = Q @ K^T")
    print("-" * 80)
    print("dL/dQ[i] = Σ_j dL/d(scores)[i,j] * K[j]")
    print("dL/dK[j] = Σ_i dL/d(scores)[i,j] * Q[i]")

    # Gradient w.r.t. Q
    for i in range(seq_len):
        for j in range(seq_len):
            for k in range(d_head):
                dL_dQ[head][i][k] += dL_d_scores_raw[i][j] * K[head][j][k]

    # Gradient w.r.t. K
    for j in range(seq_len):
        for i in range(seq_len):
            for k in range(d_head):
                dL_dK[head][j][k] += dL_d_scores_raw[i][j] * Q[head][i][k]

    print(f"  dL/dQ[{head}][0] = {format_vector(dL_dQ[head][0])}")
    print(f"  dL/dK[{head}][0] = {format_vector(dL_dK[head][0])}")

print("\n" + "=" * 80)
print("5. Backward Through Q/K/V Projections")
print("=" * 80)
print("Forward: Q[h][i] = X[i] @ W_q[h]^T")
print("         K[h][i] = X[i] @ W_k[h]^T")
print("         V[h][i] = X[i] @ W_v[h]^T")
print()
print("Backward: dL/dW_q[h] = Σ_i dL/dQ[h][i]^T ⊗ X[i]")
print("          dL/dW_k[h] = Σ_i dL/dK[h][i]^T ⊗ X[i]")
print("          dL/dW_v[h] = Σ_i dL/dV[h][i]^T ⊗ X[i]")
print("          dL/dX[i] = Σ_h (dL/dQ[h][i] @ W_q[h] + dL/dK[h][i] @ W_k[h] + dL/dV[h][i] @ W_v[h])")

# Compute weight gradients
# Forward: Q[pos] = X[pos] @ W_q, where W_q is [d_model, d_head]
# Backward: dL/dW_q = X^T @ dL/dQ = Σ_pos X[pos]^T ⊗ dL/dQ[pos]
dL_dWq = [[[0.0] * d_head for _ in range(d_model)] for _ in range(num_heads)]
dL_dWk = [[[0.0] * d_head for _ in range(d_model)] for _ in range(num_heads)]
dL_dWv = [[[0.0] * d_head for _ in range(d_model)] for _ in range(num_heads)]

for head in range(num_heads):
    for pos in range(seq_len):
        # Outer product: dL/dW[i,j] = X[pos,i] * dL/dQ[pos,j]
        for i in range(d_model):
            for j in range(d_head):
                dL_dWq[head][i][j] += X[pos][i] * dL_dQ[head][pos][j]
                dL_dWk[head][i][j] += X[pos][i] * dL_dK[head][pos][j]
                dL_dWv[head][i][j] += X[pos][i] * dL_dV[head][pos][j]

print(f"\ndL/dW_q[0][0] = {format_vector(dL_dWq[0][0])}")
print(f"dL/dW_k[0][0] = {format_vector(dL_dWk[0][0])}")
print(f"dL/dW_v[0][0] = {format_vector(dL_dWv[0][0])}")

# Compute gradients w.r.t. X (input embeddings)
# Forward: Q[pos] = X[pos] @ W_q
# Backward: dL/dX[pos] = dL/dQ[pos] @ W_q^T
dL_dX = [[0.0] * d_model for _ in range(seq_len)]

for pos in range(seq_len):
    for head in range(num_heads):
        # Backprop through Q projection: dL/dX[pos,i] = Σ_j dL/dQ[pos,j] * W_q[i,j]
        for i in range(d_model):
            for j in range(d_head):
                dL_dX[pos][i] += dL_dQ[head][pos][j] * Wq[head][i][j]

        # Backprop through K projection: dL/dX[pos,i] = Σ_j dL/dK[pos,j] * W_k[i,j]
        for i in range(d_model):
            for j in range(d_head):
                dL_dX[pos][i] += dL_dK[head][pos][j] * Wk[head][i][j]

        # Backprop through V projection: dL/dX[pos,i] = Σ_j dL/dV[pos,j] * W_v[i,j]
        for i in range(d_model):
            for j in range(d_head):
                dL_dX[pos][i] += dL_dV[head][pos][j] * Wv[head][i][j]

print(f"\ndL/dX[0] (gradient to embeddings) = {format_vector(dL_dX[0])}")

print("\n" + "=" * 80)
print("6. Backward to Embeddings")
print("=" * 80)
print("Now we have gradients w.r.t. the input embeddings X.")
print("We need to update:")
print("  - Token embeddings (E_token)")
print("  - Position embeddings (E_pos)")

# Gradients for position embeddings
# Each position contributes to its corresponding position embedding
dL_dE_pos = [[0.0] * d_model for _ in range(seq_len)]

for pos in range(seq_len):
    dL_dE_pos[pos] = dL_dX[pos][:]

print("\nGradients for position embeddings:")
for i in range(min(3, seq_len)):
    print(f"  dL/dE_pos[{i}]: {format_vector(dL_dE_pos[i])}")

# Gradients for token embeddings
# Multiple positions may use the same token, so gradients accumulate
dL_dE_token = [[0.0] * d_model for _ in range(vocab_size)]

for pos in range(seq_len):
    token_id = tokens[pos]
    for i in range(d_model):
        dL_dE_token[token_id][i] += dL_dX[pos][i]

print("\nGradients for token embeddings:")
for token_id in range(min(4, vocab_size)):
    token_name = vocab[token_id]
    print(f"  dL/dE_token[{token_id}] ({token_name}): {format_vector(dL_dE_token[token_id])}")

print("\n" + "=" * 80)
print("BACKWARD PASS COMPLETE!")
print("=" * 80)
print("\nWe've computed ACCURATE gradients for ALL parameters, including:")
print("  ✓ Softmax Jacobian in attention mechanism")
print("  ✓ Complete backprop through Q @ K^T")
print("  ✓ Causal mask gradient handling")
print("  ✓ All attention weight matrices (W_q, W_k, W_v, W_o)")
print("  ✓ Embeddings (E_token, E_pos)")
print()
print("This is the FULL, MATHEMATICALLY ACCURATE computation!")
print("No approximations were used.")
print()
print("Next step: Use these gradients with an optimizer (AdamW) to update the weights!")

# Save final gradients
print("\n" + "=" * 80)
print("Saving all gradients...")
print("=" * 80)

data = {
    'tokens': tokens,
    'X': X,
    'dL_dE_token': dL_dE_token,
    'dL_dE_pos': dL_dE_pos,
    'dL_dWq': dL_dWq,
    'dL_dWk': dL_dWk,
    'dL_dWv': dL_dWv,
    'dL_dWo': dL_dWo
}
with open('data/grad_all.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Saved to data/grad_all.pkl")
print("\nBackward pass is complete! Ready for optimization.")

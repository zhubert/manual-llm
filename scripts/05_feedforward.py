"""
Calculate feed-forward network transformation.

The FFN consists of two linear layers with GELU activation:
1. Expand: d_model (16) -> d_ff (64)
2. Activate: GELU nonlinearity
3. Project: d_ff (64) -> d_model (16)

FFN(x) = W2 @ GELU(W1 @ x + b1) + b2
"""

import random
import pickle
import math

# Set seed for reproducibility
random.seed(42)

# Model hyperparameters
D_MODEL = 16
D_FF = 64

print("=" * 80)
print("FEED-FORWARD NETWORK CALCULATION")
print("=" * 80)

# Load multi-head attention output from previous step
print("\n1. Load Multi-Head Attention Output from Previous Step")
print("-" * 80)
with open('data/multi_head.pkl', 'rb') as f:
    data = pickle.load(f)
    multi_head_output = data['multi_head_output']
    tokens = data['tokens']
    X = data['X']

seq_len = len(multi_head_output)
print(f"Sequence length: {seq_len}")
print(f"d_model: {D_MODEL}")
print(f"d_ff (hidden dimension): {D_FF}")

# Helper function to generate random matrix
def random_matrix(rows, cols, scale=0.1):
    """Generate a random matrix with values ~ N(0, scale^2)"""
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

# Helper function to generate random vector
def random_vector(size, scale=0.1):
    """Generate a random vector with values ~ N(0, scale^2)"""
    return [random.gauss(0, scale) for _ in range(size)]

# Helper function to format vector for printing
def format_vector(vec):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

# Helper function for matrix-vector multiplication
def matmul_vec(mat, vec):
    """Multiply matrix by vector: result[i] = sum(mat[i][j] * vec[j])"""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

# Helper function to add vectors
def add_vectors(v1, v2):
    """Element-wise addition of two vectors"""
    return [a + b for a, b in zip(v1, v2)]

# GELU activation function
def gelu(x):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    GELU(x) = x * Φ(x), where Φ(x) is the cumulative distribution function
    of the standard normal distribution.

    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    """
    # Using the tanh approximation
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    term = sqrt_2_over_pi * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + math.tanh(term))

# Step 2: Initialize FFN weight matrices
print("\n" + "=" * 80)
print("2. Initialize FFN Weight Matrices")
print("=" * 80)

print("\nW1: First layer weights [d_ff, d_model] = [64, 16]")
print("Projects from d_model to d_ff (expansion)")
W1 = random_matrix(D_FF, D_MODEL)

print("\nb1: First layer bias [d_ff] = [64]")
b1 = random_vector(D_FF)

print("\nW2: Second layer weights [d_model, d_ff] = [16, 64]")
print("Projects from d_ff back to d_model (projection)")
W2 = random_matrix(D_MODEL, D_FF)

print("\nb2: Second layer bias [d_model] = [16]")
b2 = random_vector(D_MODEL)

# Show first few values
print(f"\nW1[0] (first row): {format_vector(W1[0])}")
print(f"b1[:8] (first 8 values): {format_vector(b1[:8])}")
print(f"W2[0] (first row): {format_vector(W2[0][:16])}... (showing first 16 of 64)")
print(f"b2 (all values): {format_vector(b2)}")

# Step 3: Apply FFN to each position
print("\n" + "=" * 80)
print("3. Apply Feed-Forward Network")
print("=" * 80)
print("For each position, we apply:")
print("  1. Linear layer 1: hidden = W1 @ x + b1")
print("  2. GELU activation: activated = GELU(hidden)")
print("  3. Linear layer 2: output = W2 @ activated + b2")

ffn_output = []

for pos in range(seq_len):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[pos]]

    if pos == 0:  # Show detailed calculation for first position
        print(f"\n--- Detailed Calculation for Position {pos} ({token_name}) ---")

        # Step 3a: First linear layer
        print(f"\nStep 3a: First linear layer (expansion)")
        print(f"  Input: {format_vector(multi_head_output[pos])}")
        print(f"  hidden = W1 @ input + b1")
        print(f"  Shape: [64, 16] @ [16] + [64] = [64]")

        hidden = matmul_vec(W1, multi_head_output[pos])
        hidden = add_vectors(hidden, b1)

        print(f"  hidden[:8] (first 8 values): {format_vector(hidden[:8])}")

        # Step 3b: GELU activation
        print(f"\nStep 3b: GELU activation")
        print(f"  GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))")

        activated = [gelu(h) for h in hidden]

        print(f"  Example: GELU({hidden[0]:.4f}) = {activated[0]:.4f}")
        print(f"  activated[:8] (first 8 values): {format_vector(activated[:8])}")

        # Step 3c: Second linear layer
        print(f"\nStep 3c: Second linear layer (projection)")
        print(f"  output = W2 @ activated + b2")
        print(f"  Shape: [16, 64] @ [64] + [16] = [16]")

        output = matmul_vec(W2, activated)
        output = add_vectors(output, b2)

        print(f"  output: {format_vector(output)}")

        ffn_output.append(output)
    else:
        # For other positions, just compute without printing details
        hidden = matmul_vec(W1, multi_head_output[pos])
        hidden = add_vectors(hidden, b1)
        activated = [gelu(h) for h in hidden]
        output = matmul_vec(W2, activated)
        output = add_vectors(output, b2)
        ffn_output.append(output)

# Print all FFN outputs
print("\n" + "=" * 80)
print("FEED-FORWARD NETWORK OUTPUT")
print("=" * 80)
print(f"Shape: [{seq_len}, {D_MODEL}]")
print("\nThis is the output after applying the feed-forward network:")
for i, vec in enumerate(ffn_output):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    print(f"  {format_vector(vec)}  # pos {i}: {token_name}")

# Print formatted for documentation
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)

print("\n### FFN Output\n")
print("```python")
print("ffn_output = [")
for i, vec in enumerate(ffn_output):
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
    'multi_head_output': multi_head_output,
    'W1': W1,
    'b1': b1,
    'W2': W2,
    'b2': b2,
    'ffn_output': ffn_output
}
with open('data/feedforward.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/feedforward.pkl")

"""
Calculate layer normalization with residual connections.

In a transformer, we don't just use the FFN output directly. We:
1. Add a residual connection (add the attention output back)
2. Apply layer normalization to stabilize the values

This pattern (residual + layer norm) appears twice in each transformer block:
- After attention (before FFN)
- After FFN (before next block)

We're computing the second one: after the FFN.
"""

import random
import pickle
import math

# Set seed for reproducibility
random.seed(42)

# Model hyperparameters
D_MODEL = 16
EPSILON = 1e-5  # Small constant for numerical stability in layer norm

print("=" * 80)
print("LAYER NORMALIZATION CALCULATION")
print("=" * 80)

# Load FFN output from previous step
print("\n1. Load Feed-Forward Network Output from Previous Step")
print("-" * 80)
with open('data/feedforward.pkl', 'rb') as f:
    data = pickle.load(f)
    ffn_output = data['ffn_output']
    multi_head_output = data['multi_head_output']
    tokens = data['tokens']
    X = data['X']

seq_len = len(ffn_output)
print(f"Sequence length: {seq_len}")
print(f"d_model: {D_MODEL}")

# Helper function to format vector for printing
def format_vector(vec):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

# Helper function to add vectors
def add_vectors(v1, v2):
    """Element-wise addition of two vectors"""
    return [a + b for a, b in zip(v1, v2)]

# Helper function to compute mean
def mean(vec):
    """Compute mean of a vector"""
    return sum(vec) / len(vec)

# Helper function to compute variance
def variance(vec, mean_val):
    """Compute variance of a vector given its mean"""
    return sum((x - mean_val) ** 2 for x in vec) / len(vec)

# Step 2: Add residual connection
print("\n" + "=" * 80)
print("2. Add Residual Connection")
print("=" * 80)
print("The residual connection adds the attention output back to the FFN output.")
print("This prevents information loss and helps with gradient flow.\n")
print("residual_output = multi_head_output + ffn_output")

residual_output = []
for i in range(seq_len):
    residual = add_vectors(multi_head_output[i], ffn_output[i])
    residual_output.append(residual)

# Show example for position 0
token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[0]]
print(f"\n--- Example: Position 0 ({token_name}) ---")
print(f"multi_head_output[0]: {format_vector(multi_head_output[0])}")
print(f"ffn_output[0]:        {format_vector(ffn_output[0])}")
print(f"residual_output[0]:   {format_vector(residual_output[0])}")

# Step 3: Initialize layer normalization parameters
print("\n" + "=" * 80)
print("3. Initialize Layer Normalization Parameters")
print("=" * 80)

print("\nLayer normalization has two learned parameters per dimension:")
print("- gamma (scale): typically initialized to 1.0")
print("- beta (shift): typically initialized to 0.0")
print()
print("Shape: Both are [d_model] = [16]")

# Initialize gamma to 1.0 and beta to 0.0 (standard initialization)
gamma = [1.0] * D_MODEL
beta = [0.0] * D_MODEL

print(f"\ngamma (all 1.0): {format_vector(gamma)}")
print(f"beta  (all 0.0): {format_vector(beta)}")

# Step 4: Apply layer normalization
print("\n" + "=" * 80)
print("4. Apply Layer Normalization")
print("=" * 80)
print("Layer normalization formula:")
print("  1. Compute mean: μ = (1/d) Σ x_i")
print("  2. Compute variance: σ² = (1/d) Σ (x_i - μ)²")
print("  3. Normalize: x_norm = (x - μ) / √(σ² + ε)")
print("  4. Scale and shift: y = γ ⊙ x_norm + β")
print()
print("Where ε is a small constant for numerical stability (we use 1e-5)")

layer_norm_output = []

for pos in range(seq_len):
    token_name_pos = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[pos]]

    if pos == 0:  # Show detailed calculation for first position
        print(f"\n--- Detailed Calculation for Position {pos} ({token_name_pos}) ---")

        # Step 4a: Compute mean
        print(f"\nStep 4a: Compute mean")
        print(f"  Input: {format_vector(residual_output[pos])}")

        mean_val = mean(residual_output[pos])
        print(f"  μ = (1/{D_MODEL}) × Σ x_i = {mean_val:.6f}")

        # Step 4b: Compute variance
        print(f"\nStep 4b: Compute variance")
        var_val = variance(residual_output[pos], mean_val)
        print(f"  σ² = (1/{D_MODEL}) × Σ (x_i - μ)² = {var_val:.6f}")

        # Step 4c: Normalize
        print(f"\nStep 4c: Normalize")
        std_val = math.sqrt(var_val + EPSILON)
        print(f"  σ = √(σ² + ε) = √({var_val:.6f} + {EPSILON}) = {std_val:.6f}")

        normalized = [(x - mean_val) / std_val for x in residual_output[pos]]
        print(f"  x_norm = (x - μ) / σ")
        print(f"  x_norm: {format_vector(normalized)}")

        # Step 4d: Scale and shift
        print(f"\nStep 4d: Scale and shift")
        print(f"  y = γ ⊙ x_norm + β")
        print(f"  (Since γ=1.0 and β=0.0, this step doesn't change the values)")

        output = [gamma[i] * normalized[i] + beta[i] for i in range(D_MODEL)]
        print(f"  output: {format_vector(output)}")

        layer_norm_output.append(output)
    else:
        # For other positions, just compute without printing details
        mean_val = mean(residual_output[pos])
        var_val = variance(residual_output[pos], mean_val)
        std_val = math.sqrt(var_val + EPSILON)
        normalized = [(x - mean_val) / std_val for x in residual_output[pos]]
        output = [gamma[i] * normalized[i] + beta[i] for i in range(D_MODEL)]
        layer_norm_output.append(output)

# Print all layer norm outputs
print("\n" + "=" * 80)
print("LAYER NORMALIZATION OUTPUT")
print("=" * 80)
print(f"Shape: [{seq_len}, {D_MODEL}]")
print("\nThis is the output after residual connection + layer normalization:")
for i, vec in enumerate(layer_norm_output):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    print(f"  {format_vector(vec)}  # pos {i}: {token_name}")

# Print formatted for documentation
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)

print("\n### Layer Norm Output\n")
print("```python")
print("layer_norm_output = [")
for i, vec in enumerate(layer_norm_output):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    values = ", ".join([f"{v:7.4f}" for v in vec])
    print(f"  [{values}],  # pos {i}: {token_name}")
print("]")
print("```")

# Verify properties of layer norm output
print("\n" + "=" * 80)
print("VERIFICATION: Layer Norm Properties")
print("=" * 80)
print("After layer normalization, each position should have:")
print("- Mean ≈ 0.0 (due to centering)")
print("- Variance ≈ 1.0 (due to scaling)")
print()

for i in range(seq_len):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    mean_val = mean(layer_norm_output[i])
    var_val = variance(layer_norm_output[i], mean_val)
    print(f"Position {i} ({token_name:12s}): mean = {mean_val:8.6f}, variance = {var_val:.6f}")

# Save for use in next steps
print("\n" + "=" * 80)
print("Saving matrices for next steps...")
print("=" * 80)
data = {
    'X': X,
    'tokens': tokens,
    'multi_head_output': multi_head_output,
    'ffn_output': ffn_output,
    'residual_output': residual_output,
    'gamma': gamma,
    'beta': beta,
    'layer_norm_output': layer_norm_output
}
with open('data/layernorm.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/layernorm.pkl")

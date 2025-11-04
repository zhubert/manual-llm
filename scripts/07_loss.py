"""
Calculate the loss for next-token prediction.

The transformer produces hidden states (layer_norm_output), but we need
to predict tokens from our vocabulary. So we:
1. Project hidden states to vocabulary space (logits)
2. Apply softmax to get probabilities
3. Compute cross-entropy loss against the target tokens

For language modeling, the target at each position is the next token.
Input:  <BOS> I    like transformers
Target: I    like transformers <EOS>
"""

import random
import pickle
import math

# Set seed for reproducibility
random.seed(42)

# Model hyperparameters
D_MODEL = 16
VOCAB_SIZE = 6

print("=" * 80)
print("LOSS CALCULATION")
print("=" * 80)

# Load layer norm output from previous step
print("\n1. Load Layer Norm Output from Previous Step")
print("-" * 80)
with open('data/layernorm.pkl', 'rb') as f:
    data = pickle.load(f)
    layer_norm_output = data['layer_norm_output']
    tokens = data['tokens']
    X = data['X']

seq_len = len(layer_norm_output)
print(f"Sequence length: {seq_len}")
print(f"d_model: {D_MODEL}")
print(f"vocab_size: {VOCAB_SIZE}")
print(f"\nVocabulary: {['<PAD>', '<BOS>', '<EOS>', 'I', 'like', 'transformers']}")

# Helper function to generate random matrix
def random_matrix(rows, cols, scale=0.1):
    """Generate a random matrix with values ~ N(0, scale^2)"""
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]

# Helper function to format vector for printing
def format_vector(vec):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

# Helper function for matrix-vector multiplication
def matmul_vec(mat, vec):
    """Multiply matrix by vector: result[i] = sum(mat[i][j] * vec[j])"""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

# Softmax function
def softmax(logits):
    """Compute softmax: exp(x_i) / sum(exp(x_j))"""
    # Subtract max for numerical stability
    max_logit = max(logits)
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    return [x / sum_exp for x in exp_logits]

# Step 2: Initialize language modeling head (LM head)
print("\n" + "=" * 80)
print("2. Initialize Language Modeling Head (Output Projection)")
print("=" * 80)
print("\nThe LM head projects from d_model to vocab_size.")
print("W_lm: Language modeling weight matrix [vocab_size, d_model] = [6, 16]")
print("For each hidden state, this produces 6 logits (one per vocabulary token)")

W_lm = random_matrix(VOCAB_SIZE, D_MODEL)

print(f"\nW_lm[0] (weights for '<PAD>'): {format_vector(W_lm[0])}")
print(f"W_lm[1] (weights for '<BOS>'): {format_vector(W_lm[1])}")
print(f"W_lm[2] (weights for '<EOS>'): {format_vector(W_lm[2])}")

# Step 3: Compute logits
print("\n" + "=" * 80)
print("3. Compute Logits (Unnormalized Scores)")
print("=" * 80)
print("For each position, we project the hidden state to vocabulary space:")
print("  logits[pos] = W_lm @ layer_norm_output[pos]")
print("  Shape: [6, 16] @ [16] = [6]")

logits_all = []

for pos in range(seq_len):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[pos]]
    logits = matmul_vec(W_lm, layer_norm_output[pos])
    logits_all.append(logits)

    if pos == 0:  # Show detailed calculation for first position
        print(f"\n--- Detailed Calculation for Position {pos} ({token_name}) ---")
        print(f"  Input hidden state: {format_vector(layer_norm_output[pos][:8])}... (showing first 8)")
        print(f"  logits = W_lm @ hidden_state")
        print(f"  logits: {format_vector(logits)}")
        print(f"\n  Interpretation:")
        vocab = ['<PAD>', '<BOS>', '<EOS>', 'I', 'like', 'transformers']
        for i, token_logit in enumerate(logits):
            print(f"    logit[{i}] ({vocab[i]:12s}): {token_logit:7.4f}")

print("\n" + "=" * 80)
print("ALL LOGITS")
print("=" * 80)
print("These are the unnormalized scores for each token at each position:")
for i, logits in enumerate(logits_all):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    print(f"  {format_vector(logits)}  # pos {i}: {token_name}")

# Step 4: Compute probabilities with softmax
print("\n" + "=" * 80)
print("4. Compute Probabilities (Softmax)")
print("=" * 80)
print("Softmax converts logits to probabilities:")
print("  p_i = exp(logit_i) / sum(exp(logit_j))")
print("\nThis ensures all probabilities are positive and sum to 1.0")

probs_all = []

for pos in range(seq_len):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[pos]]
    probs = softmax(logits_all[pos])
    probs_all.append(probs)

    if pos == 0:  # Show detailed calculation for first position
        print(f"\n--- Detailed Calculation for Position {pos} ({token_name}) ---")
        print(f"  logits: {format_vector(logits_all[pos])}")
        print(f"  probs:  {format_vector(probs)}")
        print(f"  Sum of probs: {sum(probs):.6f} (should be 1.0)")
        print(f"\n  Interpretation (model's prediction for next token):")
        vocab = ['<PAD>', '<BOS>', '<EOS>', 'I', 'like', 'transformers']
        for i, prob in enumerate(probs):
            print(f"    P({vocab[i]:12s}) = {prob:.4f} = {prob*100:.2f}%")

print("\n" + "=" * 80)
print("ALL PROBABILITIES")
print("=" * 80)
for i, probs in enumerate(probs_all):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    print(f"  {format_vector(probs)}  # pos {i}: {token_name}")

# Step 5: Define targets (next token prediction)
print("\n" + "=" * 80)
print("5. Define Target Tokens")
print("=" * 80)
print("For language modeling, we predict the NEXT token at each position.")
print("\nInput sequence:  <BOS> I    like transformers <EOS>")
print("Token IDs:       [1,   3,   4,   5,          2]")
print("\nTarget sequence: I    like transformers <EOS>")
print("Target IDs:      [3,   4,   5,          2]")
print("\nNote: We don't predict after <EOS>, so we only compute loss for first 4 positions.")

targets = tokens[1:]  # Shift by 1 (next token at each position)
print(f"\nTargets: {targets}")

vocab = ['<PAD>', '<BOS>', '<EOS>', 'I', 'like', 'transformers']
print("\nTarget mapping:")
for i, target_id in enumerate(targets):
    current_token = vocab[tokens[i]]
    target_token = vocab[target_id]
    print(f"  Position {i} ('{current_token:12s}') -> predict '{target_token}'")

# Step 6: Compute cross-entropy loss
print("\n" + "=" * 80)
print("6. Compute Cross-Entropy Loss")
print("=" * 80)
print("Cross-entropy loss measures how well our predictions match the targets.")
print("Formula: L = -log(P(correct_token))")
print("\nLower loss = better predictions")

total_loss = 0.0
losses = []

for pos in range(len(targets)):
    target_id = targets[pos]
    prob_target = probs_all[pos][target_id]
    loss = -math.log(prob_target)
    losses.append(loss)
    total_loss += loss

    if pos == 0:  # Show detailed calculation for first position
        current_token = vocab[tokens[pos]]
        target_token = vocab[target_id]
        print(f"\n--- Position {pos}: '{current_token}' -> predict '{target_token}' ---")
        print(f"  Target ID: {target_id}")
        print(f"  Model's probability for target: P({target_token}) = {prob_target:.6f}")
        print(f"  Loss: -log({prob_target:.6f}) = {loss:.6f}")

print("\n" + "=" * 80)
print("ALL LOSSES")
print("=" * 80)
for i, loss in enumerate(losses):
    current_token = vocab[tokens[i]]
    target_token = vocab[targets[i]]
    prob = probs_all[i][targets[i]]
    print(f"  Position {i} ('{current_token}' -> '{target_token}'): loss = {loss:.6f}  (prob = {prob:.6f})")

print(f"\nTotal loss (sum): {total_loss:.6f}")
print(f"Average loss:     {total_loss / len(targets):.6f}")

# Step 7: Interpretation
print("\n" + "=" * 80)
print("7. What Does This Loss Mean?")
print("=" * 80)
print(f"Average loss: {total_loss / len(targets):.6f}")
print("\nFor comparison:")
print("  - Random guessing (uniform over 6 tokens): -log(1/6) ≈ 1.79")
print("  - Perfect prediction: -log(1.0) = 0.0")
print(f"\nOur model's loss is around {total_loss / len(targets):.2f}, which is roughly random.")
print("That makes sense — we haven't trained yet! The weights are random.")
print("\nAfter training, this loss should decrease as the model learns to predict")
print("the correct next token with higher probability.")

# Print formatted for documentation
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)

print("\n### Logits\n")
print("```python")
print("logits = [")
for i, logits in enumerate(logits_all):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    values = ", ".join([f"{v:7.4f}" for v in logits])
    print(f"  [{values}],  # pos {i}: {token_name}")
print("]")
print("```")

print("\n### Probabilities\n")
print("```python")
print("probs = [")
for i, probs in enumerate(probs_all):
    token_name = ["<PAD>", "<BOS>", "<EOS>", "I", "like", "transformers"][tokens[i]]
    values = ", ".join([f"{v:7.4f}" for v in probs])
    print(f"  [{values}],  # pos {i}: {token_name}")
print("]")
print("```")

# Save for use in backward pass
print("\n" + "=" * 80)
print("Saving data for backward pass...")
print("=" * 80)
data = {
    'X': X,
    'tokens': tokens,
    'targets': targets,
    'layer_norm_output': layer_norm_output,
    'W_lm': W_lm,
    'logits': logits_all,
    'probs': probs_all,
    'losses': losses,
    'total_loss': total_loss,
    'avg_loss': total_loss / len(targets)
}
with open('data/loss.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/loss.pkl")

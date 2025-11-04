"""
Calculate gradients of the loss with respect to logits.

This is the first step of backpropagation. We start from the loss and compute
how it changes with respect to the logits (the outputs before softmax).

For cross-entropy loss with softmax, the gradient has a beautiful closed form:
  dL/dlogits[i] = P(i) - 1{i == target}

Where P(i) is the softmax probability and 1{i == target} is 1 for the correct
class and 0 for all other classes.
"""

import pickle
import math

print("=" * 80)
print("BACKWARD PASS: LOSS GRADIENTS")
print("=" * 80)

# Load forward pass results
print("\n1. Load Forward Pass Results")
print("-" * 80)
with open('data/loss.pkl', 'rb') as f:
    data = pickle.load(f)
    tokens = data['tokens']
    targets = data['targets']
    probs = data['probs']
    logits = data['logits']
    total_loss = data['total_loss']
    X = data['X']

seq_len = len(tokens)
vocab_size = len(probs[0])

print(f"Sequence length: {seq_len}")
print(f"Vocabulary size: {vocab_size}")
print(f"Total loss: {total_loss:.6f}")
print(f"\nTargets (next tokens): {targets}")

vocab = ['<PAD>', '<BOS>', '<EOS>', 'I', 'like', 'transformers']
print("\nTarget mapping:")
for i, target_id in enumerate(targets):
    current_token = vocab[tokens[i]]
    target_token = vocab[target_id]
    print(f"  Position {i} ('{current_token:12s}') -> predict '{target_token}'")

# Helper function to format vector for printing
def format_vector(vec):
    """Format vector as string with 4 decimal places"""
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

# Step 2: The gradient formula
print("\n" + "=" * 80)
print("2. The Gradient Formula")
print("=" * 80)
print("For cross-entropy loss L = -log(P(target)) with softmax probabilities,")
print("the gradient with respect to logits has a beautiful closed form:")
print()
print("  dL/dlogits[i] = P(i) - 1{i == target}")
print()
print("Where:")
print("  - P(i) is the softmax probability for class i")
print("  - 1{i == target} is 1 if i is the correct class, 0 otherwise")
print()
print("This means:")
print("  - For the CORRECT class: gradient = P(target) - 1 (negative)")
print("  - For INCORRECT classes: gradient = P(i) - 0 = P(i) (positive)")
print()
print("Intuitively: we want to DECREASE the logit for correct class (push prob toward 1)")
print("and INCREASE logits for incorrect classes (push their probs toward 0).")
print()
print("Wait, that seems backwards! Let me clarify:")
print("  - Gradient points in direction of INCREASING loss")
print("  - We'll subtract the gradient when updating weights (gradient descent)")
print("  - So positive gradient = decrease this logit during update")
print()
print("For correct class: gradient is NEGATIVE (P - 1 < 0)")
print("  → Subtracting negative = adding → INCREASE this logit ✓")
print("For incorrect classes: gradient is POSITIVE (P > 0)")
print("  → Subtracting positive → DECREASE these logits ✓")

# Step 3: Compute gradients
print("\n" + "=" * 80)
print("3. Compute Gradients")
print("=" * 80)

dL_dlogits = []

for pos in range(len(targets)):
    target_id = targets[pos]

    # Gradient: P(i) - 1{i == target}
    grad = []
    for i in range(vocab_size):
        if i == target_id:
            grad.append(probs[pos][i] - 1.0)  # Correct class
        else:
            grad.append(probs[pos][i])  # Incorrect class

    dL_dlogits.append(grad)

    if pos == 0:  # Show detailed calculation for first position
        current_token = vocab[tokens[pos]]
        target_token = vocab[target_id]
        print(f"\n--- Detailed Calculation for Position {pos} ('{current_token}' -> '{target_token}') ---")
        print(f"  Target ID: {target_id}")
        print(f"  Probabilities: {format_vector(probs[pos])}")
        print()
        print("  Computing dL/dlogits[i] = P(i) - 1{i == target}:")
        for i in range(vocab_size):
            marker = " ← TARGET" if i == target_id else ""
            print(f"    logit[{i}] ({vocab[i]:12s}): {probs[pos][i]:.4f} - {1.0 if i == target_id else 0.0:.1f} = {grad[i]:7.4f}{marker}")
        print()
        print(f"  Gradient: {format_vector(grad)}")

# Print all gradients
print("\n" + "=" * 80)
print("ALL LOGIT GRADIENTS")
print("=" * 80)
print("These gradients tell us how to adjust the logits to reduce loss:")
for i, grad in enumerate(dL_dlogits):
    current_token = vocab[tokens[i]]
    target_token = vocab[targets[i]]
    print(f"  {format_vector(grad)}  # pos {i}: {current_token} -> {target_token}")

# Step 4: Verify the gradient
print("\n" + "=" * 80)
print("4. Verify the Gradient")
print("=" * 80)
print("The gradients should sum to 0 at each position (because softmax probabilities sum to 1).")
print()

for i in range(len(targets)):
    grad_sum = sum(dL_dlogits[i])
    current_token = vocab[tokens[i]]
    print(f"  Position {i} ('{current_token:12s}'): sum(gradients) = {grad_sum:10.8f}")

# Step 5: Intuition
print("\n" + "=" * 80)
print("5. Intuition: What Do These Gradients Mean?")
print("=" * 80)
print("Let's look at position 0: <BOS> -> I")
print()
print("Current probabilities:")
for i in range(vocab_size):
    marker = " ← TARGET" if i == targets[0] else ""
    print(f"  P({vocab[i]:12s}) = {probs[0][i]:.4f}{marker}")
print()
print("Gradients (direction to change logits to INCREASE loss):")
for i in range(vocab_size):
    marker = " ← TARGET" if i == targets[0] else ""
    direction = "increase" if dL_dlogits[0][i] < 0 else "decrease"
    print(f"  dL/dlogit[{i}] ({vocab[i]:12s}) = {dL_dlogits[0][i]:7.4f}  → to reduce loss, {direction} this logit{marker}")
print()
print("Perfect! The target ('I') has a NEGATIVE gradient, so we should INCREASE its logit.")
print("All other tokens have POSITIVE gradients, so we should DECREASE their logits.")
print("This will push the model to predict 'I' with higher probability.")

# Print formatted for documentation
print("\n" + "=" * 80)
print("FORMATTED FOR DOCUMENTATION")
print("=" * 80)

print("\n### Logit Gradients\n")
print("```python")
print("dL_dlogits = [")
for i, grad in enumerate(dL_dlogits):
    current_token = vocab[tokens[i]]
    target_token = vocab[targets[i]]
    values = ", ".join([f"{v:7.4f}" for v in grad])
    print(f"  [{values}],  # pos {i}: {current_token} -> {target_token}")
print("]")
print("```")

# Save for next backward pass step
print("\n" + "=" * 80)
print("Saving gradients for next step...")
print("=" * 80)
data = {
    'X': X,
    'tokens': tokens,
    'targets': targets,
    'probs': probs,
    'logits': logits,
    'dL_dlogits': dL_dlogits
}
with open('data/grad_loss.pkl', 'wb') as f:
    pickle.dump(data, f)
print("Saved to data/grad_loss.pkl")
print("\nNext step: backpropagate through the language modeling head (W_lm)")

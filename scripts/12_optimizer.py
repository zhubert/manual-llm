"""
AdamW Optimizer - Complete Weight Update Step

This script implements a full AdamW optimization step for all parameters.

AdamW combines:
1. Adaptive learning rates (per-parameter momentum)
2. Bias correction (accounting for initialization at zero)
3. Weight decay (L2 regularization applied directly to weights)

This is the optimizer used to train modern LLMs like GPT, LLaMA, etc.
"""

import pickle
import math

print("=" * 80)
print("ADAMW OPTIMIZER - WEIGHT UPDATE")
print("=" * 80)

# Hyperparameters
learning_rate = 0.001
beta1 = 0.9          # Exponential decay for first moment (momentum)
beta2 = 0.999        # Exponential decay for second moment (adaptive LR)
epsilon = 1e-8       # Small constant for numerical stability
weight_decay = 0.01  # L2 regularization strength
t = 1                # Time step (first update)

print(f"\nHyperparameters:")
print(f"  Learning rate (α): {learning_rate}")
print(f"  Beta1 (β₁):        {beta1}")
print(f"  Beta2 (β₂):        {beta2}")
print(f"  Epsilon (ε):       {epsilon}")
print(f"  Weight decay (λ):  {weight_decay}")
print(f"  Time step (t):     {t}")

# Load all gradients
print("\n" + "=" * 80)
print("1. Load All Gradients")
print("=" * 80)

with open('data/grad_output.pkl', 'rb') as f:
    output_data = pickle.load(f)
    dL_dW_lm = output_data['dL_dW_lm']

with open('data/grad_ffn.pkl', 'rb') as f:
    ffn_data = pickle.load(f)
    dL_dW1 = ffn_data['dL_dW1']
    dL_db1 = ffn_data['dL_db1']
    dL_dW2 = ffn_data['dL_dW2']
    dL_db2 = ffn_data['dL_db2']
    dL_dgamma = ffn_data['dL_dgamma']
    dL_dbeta = ffn_data['dL_dbeta']

with open('data/grad_all.pkl', 'rb') as f:
    attn_data = pickle.load(f)
    dL_dWq = attn_data['dL_dWq']
    dL_dWk = attn_data['dL_dWk']
    dL_dWv = attn_data['dL_dWv']
    dL_dWo = attn_data['dL_dWo']
    dL_dE_token = attn_data['dL_dE_token']
    dL_dE_pos = attn_data['dL_dE_pos']

# Load current parameter values
with open('data/embeddings.pkl', 'rb') as f:
    emb_data = pickle.load(f)
    E_token = emb_data['E_token']
    E_pos = emb_data['E_pos']

with open('data/qkv.pkl', 'rb') as f:
    qkv_data = pickle.load(f)
    Wq = qkv_data['W_Q']
    Wk = qkv_data['W_K']
    Wv = qkv_data['W_V']

with open('data/multi_head.pkl', 'rb') as f:
    mh_data = pickle.load(f)
    Wo = mh_data['W_o']

with open('data/feedforward.pkl', 'rb') as f:
    ffn_data = pickle.load(f)
    W1 = ffn_data['W1']
    b1 = ffn_data['b1']
    W2 = ffn_data['W2']
    b2 = ffn_data['b2']

with open('data/layernorm.pkl', 'rb') as f:
    ln_data = pickle.load(f)
    gamma = ln_data['gamma']
    beta = ln_data['beta']

with open('data/loss.pkl', 'rb') as f:
    loss_data = pickle.load(f)
    W_lm = loss_data['W_lm']

print("✓ Loaded all gradients and parameters")

# Helper functions
def format_vector(vec, limit=None):
    """Format vector as string with 4 decimal places"""
    if limit is not None:
        vec = vec[:limit]
    return "[" + ", ".join([f"{v:7.4f}" for v in vec]) + "]"

def initialize_moments(shape):
    """Initialize first and second moment estimates to zero"""
    if isinstance(shape, int):
        return [0.0] * shape, [0.0] * shape
    elif len(shape) == 1:
        return [0.0] * shape[0], [0.0] * shape[0]
    elif len(shape) == 2:
        m = [[0.0] * shape[1] for _ in range(shape[0])]
        v = [[0.0] * shape[1] for _ in range(shape[0])]
        return m, v
    elif len(shape) == 3:
        m = [[[0.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
        v = [[[0.0] * shape[2] for _ in range(shape[1])] for _ in range(shape[0])]
        return m, v

def adamw_update_1d(param, grad, m, v):
    """AdamW update for 1D parameter (vector)"""
    m_hat_list = []
    v_hat_list = []
    param_new = []

    for i in range(len(param)):
        # Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i]

        # Update biased second moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * (grad[i] ** 2)

        # Bias correction
        m_hat = m[i] / (1 - beta1 ** t)
        v_hat = v[i] / (1 - beta2 ** t)

        m_hat_list.append(m_hat)
        v_hat_list.append(v_hat)

        # Weight decay (applied directly to parameter, not gradient)
        # This is what makes it "AdamW" instead of "Adam"
        param_decayed = param[i] * (1 - learning_rate * weight_decay)

        # AdamW update
        param_new.append(param_decayed - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon))

    return param_new, m, v, m_hat_list, v_hat_list

def adamw_update_2d(param, grad, m, v):
    """AdamW update for 2D parameter (matrix)"""
    m_hat_all = []
    v_hat_all = []
    param_new = []

    for i in range(len(param)):
        m_hat_row = []
        v_hat_row = []
        param_row = []

        for j in range(len(param[i])):
            # Update biased first moment estimate
            m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad[i][j]

            # Update biased second moment estimate
            v[i][j] = beta2 * v[i][j] + (1 - beta2) * (grad[i][j] ** 2)

            # Bias correction
            m_hat = m[i][j] / (1 - beta1 ** t)
            v_hat = v[i][j] / (1 - beta2 ** t)

            m_hat_row.append(m_hat)
            v_hat_row.append(v_hat)

            # Weight decay
            param_decayed = param[i][j] * (1 - learning_rate * weight_decay)

            # AdamW update
            param_row.append(param_decayed - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon))

        m_hat_all.append(m_hat_row)
        v_hat_all.append(v_hat_row)
        param_new.append(param_row)

    return param_new, m, v, m_hat_all, v_hat_all

def adamw_update_3d(param, grad, m, v):
    """AdamW update for 3D parameter (e.g., multi-head weights)"""
    m_hat_all = []
    v_hat_all = []
    param_new = []

    for h in range(len(param)):
        m_hat_head = []
        v_hat_head = []
        param_head = []

        for i in range(len(param[h])):
            m_hat_row = []
            v_hat_row = []
            param_row = []

            for j in range(len(param[h][i])):
                # Update biased first moment estimate
                m[h][i][j] = beta1 * m[h][i][j] + (1 - beta1) * grad[h][i][j]

                # Update biased second moment estimate
                v[h][i][j] = beta2 * v[h][i][j] + (1 - beta2) * (grad[h][i][j] ** 2)

                # Bias correction
                m_hat = m[h][i][j] / (1 - beta1 ** t)
                v_hat = v[h][i][j] / (1 - beta2 ** t)

                m_hat_row.append(m_hat)
                v_hat_row.append(v_hat)

                # Weight decay
                param_decayed = param[h][i][j] * (1 - learning_rate * weight_decay)

                # AdamW update
                param_row.append(param_decayed - learning_rate * m_hat / (math.sqrt(v_hat) + epsilon))

            m_hat_head.append(m_hat_row)
            v_hat_head.append(v_hat_row)
            param_head.append(param_row)

        m_hat_all.append(m_hat_head)
        v_hat_all.append(v_hat_head)
        param_new.append(param_head)

    return param_new, m, v, m_hat_all, v_hat_all

# Initialize moment estimates for all parameters
print("\n" + "=" * 80)
print("2. Initialize Moment Estimates")
print("=" * 80)
print("First and second moment estimates initialized to zero (t=0)")

# Token embeddings
m_E_token, v_E_token = initialize_moments((len(E_token), len(E_token[0])))
# Position embeddings
m_E_pos, v_E_pos = initialize_moments((len(E_pos), len(E_pos[0])))
# Q, K, V weights (per head)
m_Wq, v_Wq = initialize_moments((len(Wq), len(Wq[0]), len(Wq[0][0])))
m_Wk, v_Wk = initialize_moments((len(Wk), len(Wk[0]), len(Wk[0][0])))
m_Wv, v_Wv = initialize_moments((len(Wv), len(Wv[0]), len(Wv[0][0])))
# Output projection
m_Wo, v_Wo = initialize_moments((len(Wo), len(Wo[0])))
# FFN weights and biases
m_W1, v_W1 = initialize_moments((len(W1), len(W1[0])))
m_b1, v_b1 = initialize_moments(len(b1))
m_W2, v_W2 = initialize_moments((len(W2), len(W2[0])))
m_b2, v_b2 = initialize_moments(len(b2))
# Layer norm
m_gamma, v_gamma = initialize_moments(len(gamma))
m_beta, v_beta = initialize_moments(len(beta))
# LM head
m_W_lm, v_W_lm = initialize_moments((len(W_lm), len(W_lm[0])))

print("✓ All moment estimates initialized")

# Compute bias correction factors
bias_correction1 = 1 - beta1 ** t
bias_correction2 = 1 - beta2 ** t

print(f"\nBias correction factors:")
print(f"  1 - β₁^t = 1 - {beta1}^{t} = {bias_correction1:.4f}")
print(f"  1 - β₂^t = 1 - {beta2}^{t} = {bias_correction2:.4f}")

print("\n" + "=" * 80)
print("3. Update Token Embeddings (Example Walkthrough)")
print("=" * 80)
print("Let's walk through one parameter in detail: E_token[1][0] (first element of <BOS> embedding)")

# Example: E_token[1][0]
token_idx, elem_idx = 1, 0
param_val = E_token[token_idx][elem_idx]
grad_val = dL_dE_token[token_idx][elem_idx]

print(f"\nCurrent value: θ = {param_val:.6f}")
print(f"Gradient:      g = {grad_val:.6f}")
print(f"First moment:  m₀ = {m_E_token[token_idx][elem_idx]:.6f}")
print(f"Second moment: v₀ = {v_E_token[token_idx][elem_idx]:.6f}")

# Step 1: Update biased moments
m_new = beta1 * m_E_token[token_idx][elem_idx] + (1 - beta1) * grad_val
v_new = beta2 * v_E_token[token_idx][elem_idx] + (1 - beta2) * (grad_val ** 2)

print(f"\nStep 1: Update biased moment estimates")
print(f"  m₁ = β₁·m₀ + (1-β₁)·g")
print(f"     = {beta1}·{m_E_token[token_idx][elem_idx]:.6f} + {1-beta1}·{grad_val:.6f}")
print(f"     = {m_new:.6f}")
print(f"  v₁ = β₂·v₀ + (1-β₂)·g²")
print(f"     = {beta2}·{v_E_token[token_idx][elem_idx]:.6f} + {1-beta2}·{grad_val**2:.6f}")
print(f"     = {v_new:.6f}")

# Step 2: Bias correction
m_hat = m_new / bias_correction1
v_hat = v_new / bias_correction2

print(f"\nStep 2: Bias correction")
print(f"  m̂ = m₁ / (1 - β₁^t)")
print(f"    = {m_new:.6f} / {bias_correction1:.6f}")
print(f"    = {m_hat:.6f}")
print(f"  v̂ = v₁ / (1 - β₂^t)")
print(f"    = {v_new:.6f} / {bias_correction2:.6f}")
print(f"    = {v_hat:.6f}")

# Step 3: Weight decay
param_decayed = param_val * (1 - learning_rate * weight_decay)
decay_amount = param_val - param_decayed

print(f"\nStep 3: Weight decay")
print(f"  θ_decayed = θ · (1 - α·λ)")
print(f"            = {param_val:.6f} · (1 - {learning_rate}·{weight_decay})")
print(f"            = {param_val:.6f} · {1 - learning_rate * weight_decay:.6f}")
print(f"            = {param_decayed:.6f}")
print(f"  Decay amount: {decay_amount:.6f}")

# Step 4: Adaptive learning rate and update
adaptive_lr = learning_rate / (math.sqrt(v_hat) + epsilon)
update = m_hat * adaptive_lr
param_new = param_decayed - update

print(f"\nStep 4: Compute update")
print(f"  Adaptive LR = α / (√v̂ + ε)")
print(f"              = {learning_rate} / (√{v_hat:.6f} + {epsilon})")
print(f"              = {learning_rate} / {math.sqrt(v_hat) + epsilon:.6f}")
print(f"              = {adaptive_lr:.6f}")
print(f"  Update = m̂ · adaptive_LR")
print(f"         = {m_hat:.6f} · {adaptive_lr:.6f}")
print(f"         = {update:.6f}")
print(f"  θ_new = θ_decayed - update")
print(f"        = {param_decayed:.6f} - {update:.6f}")
print(f"        = {param_new:.6f}")

print(f"\n✓ Parameter updated: {param_val:.6f} → {param_new:.6f} (change: {param_new - param_val:.6f})")

# Now update all parameters
print("\n" + "=" * 80)
print("4. Update All Parameters")
print("=" * 80)

# Token embeddings
E_token_new, m_E_token, v_E_token, _, _ = adamw_update_2d(E_token, dL_dE_token, m_E_token, v_E_token)
print(f"✓ Token embeddings updated")
print(f"  E_token[1][0]: {E_token[1][0]:.6f} → {E_token_new[1][0]:.6f}")

# Position embeddings
E_pos_new, m_E_pos, v_E_pos, _, _ = adamw_update_2d(E_pos, dL_dE_pos, m_E_pos, v_E_pos)
print(f"✓ Position embeddings updated")
print(f"  E_pos[0][0]: {E_pos[0][0]:.6f} → {E_pos_new[0][0]:.6f}")

# Attention weights
Wq_new, m_Wq, v_Wq, _, _ = adamw_update_3d(Wq, dL_dWq, m_Wq, v_Wq)
Wk_new, m_Wk, v_Wk, _, _ = adamw_update_3d(Wk, dL_dWk, m_Wk, v_Wk)
Wv_new, m_Wv, v_Wv, _, _ = adamw_update_3d(Wv, dL_dWv, m_Wv, v_Wv)
print(f"✓ Q/K/V projection weights updated")
print(f"  Wq[0][0][0]: {Wq[0][0][0]:.6f} → {Wq_new[0][0][0]:.6f}")

# Output projection
Wo_new, m_Wo, v_Wo, _, _ = adamw_update_2d(Wo, dL_dWo, m_Wo, v_Wo)
print(f"✓ Output projection weights updated")
print(f"  Wo[0][0]: {Wo[0][0]:.6f} → {Wo_new[0][0]:.6f}")

# FFN
W1_new, m_W1, v_W1, _, _ = adamw_update_2d(W1, dL_dW1, m_W1, v_W1)
b1_new, m_b1, v_b1, _, _ = adamw_update_1d(b1, dL_db1, m_b1, v_b1)
W2_new, m_W2, v_W2, _, _ = adamw_update_2d(W2, dL_dW2, m_W2, v_W2)
b2_new, m_b2, v_b2, _, _ = adamw_update_1d(b2, dL_db2, m_b2, v_b2)
print(f"✓ FFN weights and biases updated")
print(f"  W1[0][0]: {W1[0][0]:.6f} → {W1_new[0][0]:.6f}")
print(f"  b1[0]: {b1[0]:.6f} → {b1_new[0]:.6f}")

# Layer norm
gamma_new, m_gamma, v_gamma, _, _ = adamw_update_1d(gamma, dL_dgamma, m_gamma, v_gamma)
beta_new, m_beta, v_beta, _, _ = adamw_update_1d(beta, dL_dbeta, m_beta, v_beta)
print(f"✓ Layer norm parameters updated")
print(f"  gamma[0]: {gamma[0]:.6f} → {gamma_new[0]:.6f}")
print(f"  beta[0]: {beta[0]:.6f} → {beta_new[0]:.6f}")

# LM head
W_lm_new, m_W_lm, v_W_lm, _, _ = adamw_update_2d(W_lm, dL_dW_lm, m_W_lm, v_W_lm)
print(f"✓ Language modeling head updated")
print(f"  W_lm[0][0]: {W_lm[0][0]:.6f} → {W_lm_new[0][0]:.6f}")

# Calculate update statistics
print("\n" + "=" * 80)
print("5. Update Statistics")
print("=" * 80)

def count_params(param):
    """Count total number of parameters"""
    if isinstance(param[0], list):
        if isinstance(param[0][0], list):
            # 3D
            return sum(len(param[h][i]) for h in range(len(param)) for i in range(len(param[h])))
        else:
            # 2D
            return sum(len(row) for row in param)
    else:
        # 1D
        return len(param)

def avg_abs_change(old, new):
    """Average absolute change"""
    if isinstance(old[0], list):
        if isinstance(old[0][0], list):
            # 3D
            total = 0
            count = 0
            for h in range(len(old)):
                for i in range(len(old[h])):
                    for j in range(len(old[h][i])):
                        total += abs(new[h][i][j] - old[h][i][j])
                        count += 1
            return total / count
        else:
            # 2D
            total = 0
            count = 0
            for i in range(len(old)):
                for j in range(len(old[i])):
                    total += abs(new[i][j] - old[i][j])
                    count += 1
            return total / count
    else:
        # 1D
        total = sum(abs(new[i] - old[i]) for i in range(len(old)))
        return total / len(old)

total_params = (
    count_params(E_token) + count_params(E_pos) +
    count_params(Wq) + count_params(Wk) + count_params(Wv) + count_params(Wo) +
    count_params(W1) + count_params(b1) + count_params(W2) + count_params(b2) +
    count_params(gamma) + count_params(beta) + count_params(W_lm)
)

print(f"Total parameters: {total_params}")
print(f"\nAverage absolute parameter changes:")
print(f"  Token embeddings:  {avg_abs_change(E_token, E_token_new):.8f}")
print(f"  Position embeddings: {avg_abs_change(E_pos, E_pos_new):.8f}")
print(f"  Wq:                {avg_abs_change(Wq, Wq_new):.8f}")
print(f"  Wk:                {avg_abs_change(Wk, Wk_new):.8f}")
print(f"  Wv:                {avg_abs_change(Wv, Wv_new):.8f}")
print(f"  Wo:                {avg_abs_change(Wo, Wo_new):.8f}")
print(f"  W1:                {avg_abs_change(W1, W1_new):.8f}")
print(f"  b1:                {avg_abs_change(b1, b1_new):.8f}")
print(f"  W2:                {avg_abs_change(W2, W2_new):.8f}")
print(f"  b2:                {avg_abs_change(b2, b2_new):.8f}")
print(f"  gamma:             {avg_abs_change(gamma, gamma_new):.8f}")
print(f"  beta:              {avg_abs_change(beta, beta_new):.8f}")
print(f"  W_lm:              {avg_abs_change(W_lm, W_lm_new):.8f}")

print("\n" + "=" * 80)
print("TRAINING STEP COMPLETE!")
print("=" * 80)
print("\nWe've completed one full training iteration:")
print("  1. ✓ Forward pass (computed predictions)")
print("  2. ✓ Loss calculation (measured error)")
print("  3. ✓ Backward pass (computed gradients)")
print("  4. ✓ Optimization (updated weights)")
print()
print("The model has now learned from the training example.")
print("All parameters have been adjusted to reduce the loss.")
print()
print("If we ran another forward pass now, the loss would be lower!")

# Save updated parameters
print("\n" + "=" * 80)
print("Saving updated parameters...")
print("=" * 80)

data = {
    'E_token': E_token_new,
    'E_pos': E_pos_new,
    'Wq': Wq_new,
    'Wk': Wk_new,
    'Wv': Wv_new,
    'Wo': Wo_new,
    'W1': W1_new,
    'b1': b1_new,
    'W2': W2_new,
    'b2': b2_new,
    'gamma': gamma_new,
    'beta': beta_new,
    'W_lm': W_lm_new,
    # Also save moment estimates for next iteration
    'm_E_token': m_E_token,
    'v_E_token': v_E_token,
    'm_E_pos': m_E_pos,
    'v_E_pos': v_E_pos,
    'm_Wq': m_Wq,
    'v_Wq': v_Wq,
    'm_Wk': m_Wk,
    'v_Wk': v_Wk,
    'm_Wv': m_Wv,
    'v_Wv': v_Wv,
    'm_Wo': m_Wo,
    'v_Wo': v_Wo,
    'm_W1': m_W1,
    'v_W1': v_W1,
    'm_b1': m_b1,
    'v_b1': v_b1,
    'm_W2': m_W2,
    'v_W2': v_W2,
    'm_b2': m_b2,
    'v_b2': v_b2,
    'm_gamma': m_gamma,
    'v_gamma': v_gamma,
    'm_beta': m_beta,
    'v_beta': v_beta,
    'm_W_lm': m_W_lm,
    'v_W_lm': v_W_lm,
}

with open('data/optimizer.pkl', 'wb') as f:
    pickle.dump(data, f)

print("Saved to data/optimizer.pkl")
print("\nOne complete training step finished!")

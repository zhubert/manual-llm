# Attention to Detail: A Complete Training Step by Hand

An educational project that manually calculates **every single step** of training a tiny transformer model, from tokenization to weight updates. All calculations are shown step-by-step with full mathematical derivations.

## What is this?

This project walks through **every single calculation** in training a transformer-based language model:

1. **Forward Pass**: Transform text → tokens → embeddings → attention → feed-forward → predictions
2. **Backward Pass**: Calculate gradients for every parameter using backpropagation
3. **Optimization**: Update weights using the AdamW optimizer

By working through these calculations by hand (with Python for the arithmetic), you gain a deep understanding of how modern LLMs like GPT, Claude, and others actually work at the mathematical level.

## Why Manual?

This project calculates everything **by hand** (with Python for arithmetic) to build deep intuition:
- **16-dimensional embeddings** (vs GPT-3's 12,288) - small enough to visualize every value
- **2 attention heads** (vs GPT-3's 96) - simple enough to track every computation
- **1 layer** (vs GPT-3's 96) - manageable enough to complete in one session
- **Microscopic vocabulary** - focus on the math, not the data

This is the smallest practical transformer for understanding how LLMs like GPT, Claude, and others actually work.

## Architecture

Our tiny transformer:
- **Model type:** GPT-style decoder-only transformer
- **d_model:** 16 (embedding dimension)
- **Attention heads:** 2 (with d_k = d_v = 8)
- **d_ff:** 64 (feed-forward hidden dimension)
- **Layers:** 1 (single transformer block)
- **Vocabulary:** 10 tokens
- **Training text:** "the cat sat on the mat"

## Documentation

The calculations are documented step-by-step using Astro/Starlight:

```bash
cd docs
npm install
npm run dev
```

Visit http://localhost:4321/ to see the interactive documentation.

## Project Structure

```
attention-to-detail/
├── docs/                          # Astro/Starlight documentation site
│   ├── src/
│   │   ├── content/docs/          # 14 MDX pages with step-by-step calculations
│   │   └── styles/custom.css      # Mathematical display styling
│   └── package.json
├── scripts/                       # Python calculation scripts
│   ├── 01_embeddings.py           # Token and position embeddings
│   ├── 02_qkv.py                  # Query, Key, Value projections
│   ├── 03_attention.py            # Attention scores and weights
│   ├── 04_multi_head.py           # Multi-head attention combination
│   ├── 05_feedforward.py          # Feed-forward network
│   ├── 06_layernorm.py            # Layer normalization
│   ├── 07_loss.py                 # Cross-entropy loss
│   ├── 08_grad_loss.py            # Loss gradient
│   ├── 09_grad_output.py          # Output layer gradients
│   ├── 10_grad_ffn_layernorm.py   # FFN and LayerNorm gradients
│   ├── 11_grad_attention_embeddings.py  # Attention gradients
│   └── 12_optimizer.py            # AdamW optimizer updates
├── CLAUDE.md                      # Context for AI assistants
└── README.md                      # This file
```

## Related Project

This is a companion to [transformer](https://github.com/zhubert/transformer), a full PyTorch implementation of a GPT-style transformer with comprehensive documentation. That project shows you how to build and train a real model; this project shows you exactly what's happening under the hood.

## What's Completed

This project contains the **complete** training step calculation:

✅ **Forward Pass (7 steps)**
- Tokenization and embeddings
- Query, Key, Value projections
- Self-attention mechanism with softmax
- Multi-head attention combination
- Feed-forward network with GELU activation
- Layer normalization with residual connections
- Cross-entropy loss calculation

✅ **Backward Pass (5 steps)**
- Loss gradients (∂L/∂logits)
- Output layer gradients (∂L/∂W_out)
- Feed-forward gradients with chain rule
- Attention gradients with Jacobians for softmax
- Embedding gradients (∂L/∂W_E, ∂L/∂W_P)

✅ **Optimization (1 step)**
- AdamW optimizer with bias correction
- First and second moment estimates
- Weight decay and learning rate

## Learning Path

1. **Start with the docs** - `cd docs && npm run dev`
2. **Read the introduction** - Understand the architecture
3. **Follow the forward pass** - See how text becomes predictions
4. **Study the backward pass** - Learn how gradients flow through the network
5. **Examine the optimizer** - Understand how weights are updated
6. **Run the Python scripts** - Verify each calculation yourself
7. **Build intuition** - Understand why transformers work the way they do

## Key Features

🔢 **Complete Mathematical Rigor**
- Every matrix multiplication shown in full
- Jacobian matrices for complex operations (softmax, layer norm)
- Chain rule applications made explicit
- No "hand-waving" - every step justified

📊 **Visualizations**
- Color-coded matrices showing dimensions
- Step-by-step breakdowns of operations
- Interactive KaTeX math rendering
- Clear dimension tracking throughout

🐍 **Runnable Python Scripts**
- Each calculation step has a corresponding Python script
- Scripts save intermediate values for verification
- Deterministic initialization (fixed random seed)
- NumPy-based for clarity and simplicity

## Prerequisites

- Basic linear algebra (matrix multiplication, dot products)
- Understanding of neural networks (forward/backward pass, gradients)
- Familiarity with Python and NumPy (for running the scripts)
- Patience and curiosity!

## Running the Scripts

```bash
# Install dependencies
pip install numpy

# Run any script to see the calculations
python scripts/01_embeddings.py
python scripts/02_qkv.py
# ... and so on

# Intermediate values are saved in scripts/data/
```

Each script:
- Loads data from previous steps
- Performs its calculations
- Saves results for the next step
- Prints key values and shapes

## License

MIT

## Acknowledgments

This project was inspired by:
- **"Attention is All You Need"** (Vaswani et al., 2017) - The transformer architecture
- **Andrej Karpathy's educational content** - Making neural networks accessible
- **The need to truly understand** - Moving beyond treating transformers as black boxes

Special thanks to the mathematical rigor of academic papers that show every derivative, and to educational resources that prioritize understanding over implementation speed.

## What You'll Learn

After working through this project, you'll understand:
- How attention mechanisms compute relevance between tokens
- Why multi-head attention uses multiple representation subspaces
- How gradients flow through softmax and layer normalization
- Why AdamW uses both momentum and adaptive learning rates
- The actual shapes and values flowing through a transformer
- How the chain rule connects all the pieces

This is not about memorizing formulas - it's about building intuition through calculation.

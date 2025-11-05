# Claude Context: Manual LLM Calculation Project

## Project Overview

This is an educational project that manually calculates a complete training step (forward pass + backward pass + optimizer update) through a tiny transformer model. The goal is to deeply understand how LLMs work at the mathematical level by computing every single step by hand.

## Model Architecture

**Tiny GPT-style Decoder-only Transformer:**
- **d_model:** 16 (embedding dimension)
- **num_heads:** 2 (attention heads with d_k = d_v = 8)
- **d_ff:** 64 (feed-forward hidden dimension)
- **num_layers:** 1 (single transformer block to keep manageable)
- **vocab_size:** 10 tokens
- **Training text:** "the cat sat on the mat" (6 tokens)
- **Position encoding:** Learned embeddings

## Documentation Approach

This project uses **Astro/Starlight** for documentation, similar to the `/Users/zhubert/code/personal/transformer` project. The documentation is located in `docs/` and walks through the calculations step-by-step.

**Documentation structure:**
1. **Forward Pass** (7 pages): Tokenization → QKV projections → Attention → Multi-head → FFN → Layer norm → Loss
2. **Backward Pass** (5 pages): Loss gradients → Output layer → FFN → Attention → Embeddings
3. **Optimization** (2 pages): AdamW updates → Summary

### Running the docs site:
```bash
cd docs
npm install  # First time only
npm run dev  # Start dev server at http://localhost:4321/
```

## Calculation Methodology

1. **Define model architecture and initialize weights** - Random small values for weights
2. **Choose input text** - A simple sentence to train on
3. **Manually calculate forward pass** - Every matrix multiplication, activation, etc.
4. **Calculate loss** - Cross-entropy loss on the predictions
5. **Manually calculate backward pass** - Gradients for every parameter via backpropagation
6. **Apply optimizer** - AdamW weight updates with learning rate

All numerical calculations should be:
- Shown step-by-step in the MDX documentation
- Computed with Python/NumPy for accuracy
- Displayed as matrices/vectors with clear annotations
- Explained with the mathematical reasoning

## Python Scripts

The project includes 12 complete Python scripts in `scripts/`:

**Forward Pass (7 scripts):**
1. `01_embeddings.py` - Token and position embeddings
2. `02_qkv.py` - Query, Key, Value projections
3. `03_attention.py` - Attention scores and weights
4. `04_multi_head.py` - Multi-head attention combination
5. `05_feedforward.py` - Feed-forward network with GELU
6. `06_layernorm.py` - Layer normalization with residual connections
7. `07_loss.py` - Cross-entropy loss calculation

**Backward Pass (4 scripts):**
8. `08_grad_loss.py` - Loss gradients
9. `09_grad_output.py` - Output layer gradients
10. `10_grad_ffn_layernorm.py` - FFN and LayerNorm gradients
11. `11_grad_attention_embeddings.py` - Attention and embedding gradients

**Optimization (1 script):**
12. `12_optimizer.py` - AdamW optimizer with bias correction

Each script:
- Uses deterministic initialization (fixed random seed)
- Loads intermediate values from previous steps
- Performs all calculations with NumPy
- Saves results to `scripts/data/` for the next step
- Prints key values and shapes for verification

## Related Project

This is a companion to the [transformer](https://github.com/zhubert/transformer) repository, which provides:
- A full PyTorch implementation of a GPT-style transformer
- Comprehensive documentation with similar educational approach
- Training details and practical implementation patterns
- Architecture decisions and design rationale

The `transformer` repo shows you how to build and train a real model; this project shows you exactly what's happening mathematically under the hood.

## Custom CSS Classes

The docs use custom CSS classes for displaying calculations:

- `.formula` - Mathematical formulas
- `.matrix` - Matrix displays
- `.calc-step` - Individual calculation steps
- `.value-highlight` - Highlighting specific values

See `docs/src/styles/custom.css` for all available styles.

## Working with This Project

The core calculations are complete. When helping with this project:

**For Explanations:**
- Explain existing calculations in different ways
- Answer questions about the mathematics
- Clarify why certain approaches were taken
- Help understand the chain rule applications

**For Improvements:**
- Enhance documentation clarity
- Add additional visualizations or examples
- Improve code comments and docstrings
- Fix any errors or typos

**For Extensions (if requested):**
- Add more tokens or sequences
- Implement additional optimizer algorithms (SGD, Adam)
- Add more transformer components (e.g., encoder-decoder attention)
- Create visualizations of attention patterns

**Guiding Principles:**
1. **Maintain pedagogical focus** - Everything should teach, not just work
2. **Show all work** - Don't skip steps; show every intermediate calculation
3. **Be mathematically rigorous** - Justify every operation
4. **Keep it educational** - Explain the "why" not just the "what"

## Current Status

**The project is COMPLETE!** All calculations, documentation, and scripts are finished.

### Completed Work:

✅ **Documentation (14 MDX pages)**
- Introduction and architecture overview
- 7 forward pass pages with full calculations
- 5 backward pass pages with complete gradient derivations
- 1 optimizer page with AdamW implementation
- 1 summary page

✅ **Python Scripts (12 complete)**
- All forward pass calculations (embeddings → loss)
- All backward pass gradients (loss → embeddings)
- Complete AdamW optimizer implementation
- Deterministic weight initialization
- Intermediate value saving/loading

✅ **Mathematical Rigor**
- Every matrix multiplication shown step-by-step
- Jacobian matrices for softmax and layer normalization
- Complete chain rule applications
- Dimension tracking throughout
- No steps skipped or hand-waved

✅ **Educational Features**
- KaTeX rendering for all equations
- Color-coded matrices with dimension labels
- Step-by-step breakdowns of complex operations
- Custom CSS for clear mathematical displays

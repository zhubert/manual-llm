# Claude Context: Manual LLM Calculation Project

## Project Overview

This is an educational project that manually calculates a complete training step (forward pass + backward pass + optimizer update) through a tiny transformer model. The goal is to deeply understand how LLMs work at the mathematical level by computing every single step by hand.

## Model Architecture

**Tiny GPT-style Decoder-only Transformer:**
- **d_model:** 16 (embedding dimension)
- **num_heads:** 2 (attention heads)
- **d_ff:** 64 (feed-forward hidden dimension)
- **num_layers:** 1 (single transformer block to keep manageable)
- **vocab_size:** Small vocabulary (TBD - just enough for our example sentence)
- **Position encoding:** Learned embeddings (or ALiBi - TBD)

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

## Python Helper Scripts

There should be Python scripts (in `scripts/` or root) that:
- Initialize the model weights deterministically (with a fixed seed)
- Compute all forward pass values
- Compute all backward pass gradients
- Perform optimizer updates
- Generate the exact numerical values that appear in the docs

This ensures consistency and allows regenerating values if we change the architecture.

## Key Reference

The full transformer implementation is at `/Users/zhubert/code/personal/transformer` - refer to it for:
- Implementation patterns
- Architecture decisions
- Training details
- Documentation style

## Custom CSS Classes

The docs use custom CSS classes for displaying calculations:

- `.formula` - Mathematical formulas
- `.matrix` - Matrix displays
- `.calc-step` - Individual calculation steps
- `.value-highlight` - Highlighting specific values

See `docs/src/styles/custom.css` for all available styles.

## Working with This Project

When helping with this project:
1. **User wants manual control** - They want to work through calculations themselves, only use automation where requested
2. **Document as we go** - Create MDX pages that explain each step clearly
3. **Be pedagogical** - This is educational, so explain the "why" not just the "what"
4. **Show all work** - Don't skip steps; show every intermediate calculation
5. **Use the transformer project as a reference** - Architecture and documentation style should match

## Current Status

- ✅ Astro/Starlight site structure complete
- ✅ All placeholder pages created
- ⏳ Need to define exact model architecture
- ⏳ Need to choose input text
- ⏳ Need to initialize weights
- ⏳ Begin step-by-step calculations

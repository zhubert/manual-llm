# The Worst LLM: Manual Training Calculation

An educational project that manually calculates a complete training step through a tiny transformer model, from tokenization to weight updates.

## What is this?

This project walks through **every single calculation** in training a transformer-based language model:

1. **Forward Pass**: Transform text → tokens → embeddings → attention → feed-forward → predictions
2. **Backward Pass**: Calculate gradients for every parameter using backpropagation
3. **Optimization**: Update weights using the AdamW optimizer

By working through these calculations by hand (with Python for the arithmetic), you gain a deep understanding of how modern LLMs like GPT, Claude, and others actually work at the mathematical level.

## Why "The Worst LLM"?

Because it's intentionally tiny and impractical:
- **16-dimensional embeddings** (GPT-3 uses 12,288)
- **2 attention heads** (GPT-3 uses 96)
- **1 layer** (GPT-3 uses 96)
- **Microscopic vocabulary**

It's the worst possible LLM for any real use... but the **best** for learning!

## Architecture

- **Model type:** GPT-style decoder-only transformer
- **d_model:** 16 (embedding dimension)
- **Attention heads:** 2
- **d_ff:** 64 (feed-forward hidden dimension)
- **Layers:** 1
- **Vocabulary:** Small (just enough for our example)

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
the-worst-llm/
├── docs/                  # Astro/Starlight documentation site
│   ├── src/
│   │   ├── content/docs/  # MDX pages with calculations
│   │   └── styles/        # Custom CSS
│   └── package.json
├── scripts/               # Python scripts for calculations
├── CLAUDE.md             # Context for Claude Code
└── README.md             # This file
```

## Related Project

This is a companion to [transformer](https://github.com/zhubert/transformer), a full PyTorch implementation of a GPT-style transformer with comprehensive documentation. That project shows you how to build and train a real model; this project shows you exactly what's happening under the hood.

## Learning Path

1. **Start with the docs** - `cd docs && npm run dev`
2. **Follow the calculations** - Work through each step methodically
3. **Run the Python scripts** - Verify the calculations yourself
4. **Build intuition** - Understand why transformers work the way they do

## Prerequisites

- Basic linear algebra (matrix multiplication, dot products)
- Understanding of neural networks (forward/backward pass, gradients)
- Familiarity with Python and NumPy
- Patience and curiosity!

## License

MIT

## Acknowledgments

Inspired by:
- "Attention is All You Need" (Vaswani et al., 2017)
- Andrej Karpathy's educational content
- The need to truly understand what we're building

# KaTeX Math Notation Usage Guide

KaTeX is now enabled for all documentation pages! You can use LaTeX-style math notation in your MDX files.

## Inline Math

Use single dollar signs `$...$` for inline math:

```markdown
The dimension per head is $d_k = \frac{d_{model}}{num\_heads} = \frac{16}{2} = 8$
```

Renders as: The dimension per head is $d_k = \frac{d_{model}}{num\_heads} = \frac{16}{2} = 8$

## Block Math (Display Mode)

Use double dollar signs `$$...$$` for centered block equations:

```markdown
$$
Q = XW_Q
$$
```

Renders as:
$$
Q = XW_Q
$$

## Common Examples

### Matrix Multiplication
```markdown
$$
\text{result}_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$
```

$$
\text{result}_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}
$$

### Attention Score
```markdown
$$
\text{score}_{ij} = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}
$$
```

$$
\text{score}_{ij} = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}
$$

### Softmax
```markdown
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$
```

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

### Matrix Shapes
```markdown
$$
X \in \mathbb{R}^{5 \times 16}, \quad W_Q \in \mathbb{R}^{16 \times 8}, \quad Q \in \mathbb{R}^{5 \times 8}
$$
```

$$
X \in \mathbb{R}^{5 \times 16}, \quad W_Q \in \mathbb{R}^{16 \times 8}, \quad Q \in \mathbb{R}^{5 \times 8}
$$

### Vectors
```markdown
$$
\vec{x} = [x_1, x_2, \ldots, x_n]
$$
```

$$
\vec{x} = [x_1, x_2, \ldots, x_n]
$$

### Aligned Equations
```markdown
$$
\begin{aligned}
Q[0][0][0] &= (0.1473 \times -0.0144) + (0.1281 \times -0.0217) + \ldots \\
&= -0.0021 - 0.0028 + \ldots \\
&= -0.0783
\end{aligned}
$$
```

$$
\begin{aligned}
Q[0][0][0] &= (0.1473 \times -0.0144) + (0.1281 \times -0.0217) + \ldots \\
&= -0.0021 - 0.0028 + \ldots \\
&= -0.0783
\end{aligned}
$$

## Useful Symbols

- Subscripts: `x_i` → $x_i$
- Superscripts: `x^2` → $x^2$
- Fractions: `\frac{a}{b}` → $\frac{a}{b}$
- Square root: `\sqrt{x}` → $\sqrt{x}$
- Summation: `\sum_{i=1}^{n}` → $\sum_{i=1}^{n}$
- Product: `\prod_{i=1}^{n}` → $\prod_{i=1}^{n}$
- Greek letters: `\alpha, \beta, \theta` → $\alpha, \beta, \theta$
- Vectors: `\vec{x}` → $\vec{x}$
- Matrices: `\mathbf{X}` → $\mathbf{X}$
- Transpose: `A^T` → $A^T$
- Real numbers: `\mathbb{R}` → $\mathbb{R}$
- Dot product: `\cdot` → $\cdot$
- Times: `\times` → $\times$

## Documentation

For more details, see:
- [KaTeX Supported Functions](https://katex.org/docs/supported.html)
- [KaTeX Support Table](https://katex.org/docs/support_table.html)

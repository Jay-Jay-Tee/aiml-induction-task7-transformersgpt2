README: GPT-2 Transformer

🌻 Overview

This submission explains how a decoder-only Transformer (GPT-2) processes tokens and includes two core components implemented as independent modules:

Control Flow Diagram (CFD): Hand-drawn diagram showing the end-to-end flow of data through GPT-2 while processing tokens.

Multi-Head Masked Self-Attention module: Implements causal attention (no future token access).

Positional Encoding module: Adds learned positional information to token embeddings (GPT-2 style).

GPT-2 is decoder-only, meaning it uses:
Masked self-attention (causal)
No encoder / no cross-attention
Autoregressive next-token prediction

👑 Deliverables

1) Control Flow Diagram (CFD)

A pen-and-paper diagram showing how GPT-2 processes tokens:

Pipeline:

Input text → tokenization → token IDs
Token embedding lookup
Positional embedding lookup
Add embeddings (token + position)
Pass through N Transformer blocks
Pre-LayerNorm
Masked Multi-Head Self-Attention + residual addition
Pre-LayerNorm
Feed Forward Network (MLP) + residual addition
Final LayerNorm
Unembedding / LM Head -> logits
Softmax -> next token probabilities
Append next token → repeat (autoregressive loop)

Key CFD notes:

Mask prevents attention to future tokens (upper triangle masked).
Each block refines token representations (“multiple thinking steps”).
Only the last position’s logits are used for next-token generation.

2) Program: Positional Encoding (GPT-2 Learned Positions)

What it does

Transformers have no inherent sense of order. This module adds learned positional embeddings to token embeddings so the model knows which token came first/second/etc.

Core idea

For sequence length T:
Get token embeddings E_token with shape (T, d_model)
Get positional embeddings E_pos[:T] with shape (T, d_model)
Output: X = E_token + E_pos

3) Program: Masked Multi-Head Self-Attention (GPT-2)

What it does

For each token, attention computes which previous tokens matter, and gathers information from them while ensuring no future tokens are visible (causal mask).

Steps inside the module

Given input X of shape (T, d_model):
Project into Q, K, V
Q = X dot WQ, K = X dot WK, V = X dot WV
Split into heads
Reshape to (T, num_heads, d_head)

For each head

Compute similarity: scores = (Q dot K^T)/sqrt(d_head) → shape (T, T)
Apply causal mask (upper triangle set to very negative)
Softmax to get attention weights
Weighted sum: attention dot V -> shape (T, d_head)
Concatenate heads
Shape back to (T, d_model)

Output projection

Multiply by W_O to mix head outputs into final attention output

NOTE: Why we mask the upper triangle

scores[i, j] means “token i attends to token j”.
If j > i, that token is in the future -> must be blocked to keep autoregressive generation valid.

✌️ How to run (if applicable)

If these modules are standalone educational implementations, you can include a simple test call (optional).

Example:
Create dummy token_embeddings with shape (T, d_model)
Add positional encoding
Pass through multi-head attention
(Training is not required for this induction deliverable.)

🚀 Notes / Assumptions

Implementations are simplified for learning and readability.
In real GPT-2, embeddings and attention weights and other weight matrices are trained via backpropagation.
Softmax should ideally be numerically stable (practical models use a stabilized softmax).
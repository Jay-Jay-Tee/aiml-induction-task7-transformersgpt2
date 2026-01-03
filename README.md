# GPT-2 Transformer (Decoder-Only)  
**GDSC AI/ML Inductions – Advanced Task 1**

![task](https://img.shields.io/badge/Task-Advanced%201-blue)
![type](https://img.shields.io/badge/Type-Transformer%20Architecture-brightgreen)
![model](https://img.shields.io/badge/Model-GPT--2%20(Decoder--Only)-orange)

---

## 🌻 Overview

This submission explains how a **decoder-only Transformer (GPT-2)** processes tokens and includes two core components implemented as independent modules:

- **Control Flow Diagram (CFD):** Hand-drawn diagram showing the end-to-end flow of data through GPT-2 during token processing  
- **Masked Multi-Head Self-Attention module:** Implements causal attention (no future token access)  
- **Positional Encoding module:** Adds learned positional information to token embeddings (GPT-2 style)

GPT-2 is **decoder-only**, meaning it uses:
- Masked self-attention (causal)
- No encoder / no cross-attention
- Autoregressive next-token prediction

---

## 👑 Deliverables

### 1) Control Flow Diagram (CFD)

A pen-and-paper diagram illustrating how GPT-2 processes tokens.

![TransformerP1](https://github.com/user-attachments/assets/a1a960de-58b3-4d0b-89e2-2ebb1d7333cc)


![TransformerP2](https://github.com/user-attachments/assets/c68a3197-85c1-4ea1-83c4-9a207022311a)


#### Pipeline

- Input text → tokenization → token IDs  
- Token embedding lookup  
- Positional embedding lookup  
- Add embeddings (token + position)  
- Pass through **N Transformer blocks**
  - Pre-LayerNorm  
  - Masked Multi-Head Self-Attention + residual addition  
  - Pre-LayerNorm  
  - Feed Forward Network (MLP) + residual addition  
- Final LayerNorm  
- Unembedding / LM Head → logits  
- Softmax → next-token probabilities  
- Append next token → repeat (autoregressive loop)

#### Key CFD Notes

- The mask prevents attention to future tokens (upper triangle masked)
- Each block refines token representations (“multiple thinking steps”)
- Only the **last position’s logits** are used for next-token generation

---

## 2) Program: Positional Encoding (GPT-2 Learned Positions)

### What It Does

Transformers have no inherent sense of order. This module adds **learned positional embeddings** to token embeddings so the model knows token positions in a sequence.

### Core Idea

For sequence length **T**:

- Token embeddings: `E_token` with shape `(T, d_model)`
- Positional embeddings: `E_pos[:T]` with shape `(T, d_model)`
- Output representation:

    X = E_token + E_pos

This follows the **GPT-2 learned positional embedding** approach rather than sinusoidal encodings.

---

## 3) Program: Masked Multi-Head Self-Attention (GPT-2)

### What It Does

For each token, attention determines which **previous tokens** matter and aggregates information from them, while ensuring **future tokens are not visible** via a causal mask.

---

### Steps Inside the Module

Given input `X` of shape `(T, d_model)`:

- Project into queries, keys, values  
  - Q = X · W_Q  
  - K = X · W_K  
  - V = X · W_V  
- Split into multiple heads  
- Reshape to `(T, num_heads, d_head)`

---

### For Each Head

- Compute similarity scores:

    scores = (Q · Kᵀ) / √d_head   → shape `(T, T)`

- Apply **causal mask** (upper triangle set to very negative values)
- Apply softmax to obtain attention weights
- Compute weighted sum:

    attention · V  → shape `(T, d_head)`

---

### Head Combination & Output Projection

- Concatenate all heads  
- Reshape back to `(T, d_model)`  
- Apply output projection:

    output = concatenated_heads · W_O

---

### NOTE: Why the Upper Triangle Is Masked

- `scores[i, j]` represents “token *i* attends to token *j*”
- If `j > i`, that token lies in the **future**
- Allowing this would break autoregressive generation
- Masking enforces strict left-to-right prediction

---

## ✌️ How to Run (If Applicable)

If these modules are implemented as standalone educational components, a simple test call may be used:

- Create dummy `token_embeddings` with shape `(T, d_model)`
- Add positional encoding
- Pass through masked multi-head self-attention

Training is **not required** for this induction deliverable.

---

## 🚀 Notes / Assumptions

- Implementations are simplified for learning and readability
- In real GPT-2, embeddings and attention weights are learned via backpropagation
- Softmax should ideally be numerically stable (production models use stabilized softmax variants)

---

## 📌 Summary

This submission demonstrates a conceptual and modular understanding of **GPT-2’s decoder-only Transformer architecture**, focusing on:
- Autoregressive token processing
- Learned positional embeddings
- Causal masked self-attention


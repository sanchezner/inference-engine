# inference engine

## what this is

A from-scratch inference runtime I'm building to get a deeper understanding of inference engines.

## log:

**2026-10-03 ~ Forward pass matches the oracle** [[blog post](https://sanchezner.com/2026/10/03/a-naive-forward-pass)]
Wrote the GPT-2-small forward pass and checked it against the Hugging Face `gpt2` checkpoint on "The capital of France is". Token and position embeddings feed 12 pre-norm blocks, each doing causal attention and a tanh-GELU MLP, and the last position is multiplied by the tied token embedding to score the next token.
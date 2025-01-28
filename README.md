# Masked Language Model with BERT

This project explores the use of a Masked Language Model (MLM) built on BERT (Bidirectional Encoder Representations from Transformers) to predict masked words in text sequences and visualize attention patterns in BERT's multi-headed attention mechanism.

## Overview

Masked Language Modeling is a pre-training task for language models like BERT, where the model predicts masked words based on the surrounding context. This project involves:
1. **Using Hugging Face's transformers library** to predict masked words with BERT.
2. **Visualizing attention scores** from BERT's 144 attention heads to gain insights into its language understanding.

## Key Features

- **Masked Word Prediction**: Replace a word in a sentence with `[MASK]`, and the model predicts possible replacements.
- **Attention Visualization**: Generate attention diagrams for each of BERT's attention heads to explore relationships between tokens.
- **Analysis of Attention Heads**: Investigate the roles of specific attention heads in understanding natural language.

## Requirements

- Python 3.7 or higher
- Install dependencies using the provided `requirements.txt` file:
  ```bash
  pip3 install -r requirements.txt

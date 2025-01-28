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
## Generating Attention Diagrams
The script also visualizes attention patterns across tokens in the input text. Each diagram represents attention weights for specific layers and heads of the BERT model. For instance:

**Input:**  
`"The [MASK] sat on the mat."`

**Output:**  
Attention diagrams are generated and saved as image files:  
- `Attention_Layer1_Head1.png`  
- `Attention_Layer1_Head2.png`  
- ...  

These diagrams show the relationships between tokens based on attention weights.

---

## Analysis of Attention Heads
From the analysis, attention heads exhibit distinct behaviors. For example:

### Layer 5, Head 8:
Focuses on relationships between nouns and their adjectives.

**Example Sentences:**  
1. `"The [MASK] car drove down the empty street."`  
   - **Prediction:** `"red"`, `"blue"`, `"fast"`.  
   - The head attends to "car" and associates it with likely adjectives.  

2. `"She bought a [MASK] dress."`  
   - **Prediction:** `"beautiful"`, `"red"`, `"new"`.  
   - The head attends to "dress" and connects it with descriptive adjectives.  

---

### Layer 7, Head 3:
Appears to capture verb-subject relationships.

**Example Sentences:**  
1. `"The dog [MASK] quickly."`  
   - **Prediction:** `"ran"`, `"jumped"`, `"barked"`.  
   - The head connects "dog" to likely actions.  

2. `"A bird [MASK] into the room."`  
   - **Prediction:** `"flew"`, `"came"`, `"glided"`.  
   - The head links "bird" with typical movements.

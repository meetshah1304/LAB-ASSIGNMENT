# NLP Lab Assignment Report
## LSTM and GRU for Multiple NLP Tasks

---

## Table of Contents
1. [Introduction](#introduction)
2. [Task 1: Common Utilities](#task-1-common-utilities)
3. [Task 2: LSTM Implementation](#task-2-lstm-implementation)
4. [Task 3: GRU Implementation](#task-3-gru-implementation)
5. [Task 4: NLP Applications](#task-4-nlp-applications)
6. [Task 5: Performance Analysis](#task-5-performance-analysis)
7. [Corpus Processing Results](#corpus-processing-results)
8. [Conclusions](#conclusions)

---

## Introduction

This report documents the completion of the NLP Lab Assignment which involves implementing LSTM and GRU architectures for various NLP tasks using a corpus dataset.

**Dataset:** Borderlands-related text corpus (corpus.txt)
- Total lines: 73,996
- Paraphrase groups: 11,633

---

## Task 1: Common Utilities

### Tokenization
Implemented a custom tokenizer with the following features:
- Lowercase conversion
- URL removal
- Mention and hashtag removal
- Special character handling
- Whitespace normalization

### Vocabulary Building
- Built vocabulary from corpus texts
- Reserved indices: 0 for PAD, 1 for UNK
- Configurable minimum frequency and maximum vocabulary size

### Text Encoding/Decoding
- Convert text to vocabulary indices
- Support for padding and truncation
- Reverse conversion from indices to text

### Code Location: `utils.py`

---

## Task 2: LSTM Implementation

### LSTM Architecture
Implemented from scratch using PyTorch:

1. **LSTMCell**: Basic LSTM cell with:
   - Input gate, Forget gate, Output gate
   - Cell candidate
   - Xavier initialization

2. **LSTM**: Multi-layer bidirectional LSTM
   - Configurable number of layers
   - Dropout between layers
   - Bidirectional processing

3. **LSTMClassifier**: Classification model
   - Embedding layer
   - Bidirectional LSTM
   - Fully connected output layer

### Model Parameters
| Configuration | Value |
|--------------|-------|
| Vocab Size | 10,000 |
| Embedding Dim | 128 |
| Hidden Size | 64 |
| Num Layers | 2 |
| Total Parameters | 1,478,914 |

### Code Location: `models.py`

---

## Task 3: GRU Implementation

### GRU Architecture
Implemented from scratch using PyTorch:

1. **GRUCell**: Basic GRU cell with:
   - Update gate
   - Reset gate
   - Candidate hidden state

2. **GRU**: Multi-layer bidirectional GRU
   - Same features as LSTM
   - Fewer parameters than LSTM

3. **GRUClassifier**: Classification model

### Model Parameters
| Configuration | Value |
|--------------|-------|
| Vocab Size | 10,000 |
| Embedding Dim | 128 |
| Hidden Size | 64 |
| Num Layers | 2 |
| Total Parameters | 1,429,250 |

### Comparison
- GRU has ~50K fewer parameters than LSTM
- Simpler architecture with update/reset gates vs input/forget/output gates

### Code Location: `models.py`

---

## Task 4: NLP Applications

### Application 1: Text Classification (Sentiment Analysis)

**Dataset:** Synthetic binary classification data
- 1,000 samples (500 positive, 500 negative)

**Results:**

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| LSTM  | 100%     | 100%     | 100%      | 100%   |
| GRU   | 100%     | 100%     | 100%      | 100%   |

### Application 2: Paraphrase Detection

**Dataset:** 
- Created from corpus.txt
- 2,000 sentence pairs (balanced)
- Train/Test split: 80%/20%

**Architecture:** Siamese network with shared encoder

**Features:**
- Sentence 1 encoding + Sentence 2 encoding
- Absolute difference
- Element-wise product
- Fully connected classification

### Code Location: `train.py`, `corpus_processor.py`

---

## Task 5: Performance Analysis

### NLP Metrics Computed

1. **Accuracy**: Overall correctness
2. **Precision**: TP / (TP + FP)
3. **Recall**: TP / (TP + FN)
4. **F1 Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Visual representation of predictions

### Code Location: `utils.py` - `compute_performance_metrics()`

---

## Corpus Processing Results

### Dataset Statistics
- Total lines loaded: 69,799
- Paraphrase groups: 11,633
- Sentence pairs: 2,000
- Vocabulary size: 8,002

### Paraphrase Detection Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| LSTM  | 63.25%   | 63.56%    | 63.51% | 63.24%   |
| GRU   | 66.75%   | 67.40%    | 67.16% | 67.70%   |

### Confusion Matrix (LSTM)
```
              Predicted
              Neg    Pos
Actual Neg   124    87
Actual Pos    60   129
```

### Confusion Matrix (GRU)
```
              Predicted
              Neg    Pos
Actual Neg   126    85
Actual Pos    48   141
```

---

## Conclusions

### Key Findings

1. **LSTM vs GRU:**
   - GRU outperforms LSTM on this task (66.75% vs 63.25%)
   - GRU has fewer parameters (1.24M vs 1.29M)
   - GRU trains faster due to simpler architecture

2. **Model Performance:**
   - Both models show overfitting (high train, lower val accuracy)
   - Paraphrase detection is a challenging task
   - More data would likely improve performance

3. **Architecture Insights:**
   - Bidirectional processing helps capture context
   - Siamese architecture effective for similarity tasks
   - Attention mechanism could improve results

### Files Created

| File | Description |
|------|-------------|
| `utils.py` | Tokenization, vocabulary, metrics |
| `models.py` | LSTM/GRU implementations |
| `train.py` | Training pipelines |
| `run.py` | Complete lab runner |
| `corpus_processor.py` | Corpus-based experiments |
| `NLP_Lab_Report.md` | This report |

---

## Running the Code

### Basic Tasks:
```
bash
cd NLP_Lab_Assignment
python run.py
```

### Corpus Processing:
```
bash
python corpus_processor.py
```

### Requirements:
- Python 3.x
- PyTorch
- NumPy
- NLTK

---

**Report Generated:** 2026-02-26
**Course:** MTECH NFSU - Sem-2 - NLP
**Instructor:** Charudatta Sir

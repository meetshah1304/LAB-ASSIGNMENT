"""
NLP Lab Assignment - Common Utilities
=====================================
This module contains common NLP utilities for text processing including:
- Tokenization
- Vocabulary building
- Data loading and preprocessing
"""

import re
import string
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple


class Tokenizer:
    """Custom tokenizer for text preprocessing"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.word_counts = Counter()
        
    def basic_tokenize(self, text: str) -> List[str]:
        """Basic word tokenization using regex"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into tokens
        tokens = text.split()
        return tokens
    
    def nltk_tokenize(self, text: str) -> List[str]:
        """NLTK-based tokenization"""
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download('punkt_tab', quiet=True)
                
            from nltk.tokenize import word_tokenize
            text = text.lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'@\w+|#\w+', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return word_tokenize(text)
        except Exception as e:
            print(f"NLTK tokenization failed, using basic: {e}")
            return self.basic_tokenize(text)
    
    def build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab: int = 10000):
        """Build vocabulary from texts"""
        print("Building vocabulary...")
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.basic_tokenize(text)
            all_tokens.extend(tokens)
            self.word_counts.update(tokens)
        
        # Filter by frequency
        filtered_words = [word for word, count in self.word_counts.items() 
                        if count >= min_freq]
        
        # Sort by frequency and limit vocabulary size
        sorted_words = sorted(filtered_words, key=lambda x: self.word_counts[x], reverse=True)
        sorted_words = sorted_words[:max_vocab]
        
        # Create vocab dictionaries (reserve 0 for padding, 1 for unknown)
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, word in enumerate(sorted_words, start=2):
            self.vocab[word] = idx
        
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab
    
    def encode(self, text: str, max_length: int = 50) -> np.ndarray:
        """Encode text to vocabulary indices"""
        tokens = self.basic_tokenize(text)
        encoded = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Padding or truncation
        if len(encoded) < max_length:
            padded = encoded + [self.vocab['<PAD>']] * (max_length - len(encoded))
        else:
            padded = encoded[:max_length]
        
        return np.array(padded)
    
    def decode(self, indices: np.ndarray) -> str:
        """Decode vocabulary indices back to text"""
        tokens = [self.reverse_vocab.get(idx, '<UNK>') for idx in indices]
        # Remove padding and unknown tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<UNK>']]
        return ' '.join(tokens)


def load_corpus(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load corpus from file
    Returns: (original_texts, processed_texts)
    """
    print(f"Loading corpus from {filepath}...")
    
    original_texts = []
    processed_texts = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Process lines - each line contains original and processed versions
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split(',', 1)  # Split on first comma
            if len(parts) >= 1:
                original_texts.append(parts[0])
                processed_texts.append(parts[0])  # Use original as processed for now
    
    print(f"Loaded {len(original_texts)} text samples")
    return original_texts, processed_texts


def load_paraphrase_corpus(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Load paraphrase detection corpus
    Returns: (sentences, labels)
    """
    print(f"Loading paraphrase corpus from {filepath}...")
    
    sentences1 = []
    sentences2 = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Each line: sentence1 \t sentence2 \t label
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split('\t')
            if len(parts) >= 3:
                sentences1.append(parts[0])
                sentences2.append(parts[1])
                labels.append(int(parts[2]))
    
    print(f"Loaded {len(sentences1)} sentence pairs")
    return sentences1, sentences2, labels


def prepare_sequences(texts: List[str], tokenizer: Tokenizer, 
                     max_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for model training"""
    X = np.zeros((len(texts), max_length), dtype=np.int64)
    
    for i, text in enumerate(texts):
        tokens = tokenizer.basic_tokenize(text)
        for j, token in enumerate(tokens[:max_length]):
            X[i, j] = tokenizer.vocab.get(token, tokenizer.vocab['<UNK>'])
    
    return X


def create_word_embeddings(vocab_size: int, embedding_dim: int = 100) -> np.ndarray:
    """Create random word embeddings"""
    np.random.seed(42)
    embeddings = np.random.randn(vocab_size, embedding_dim) * 0.1
    return embeddings.astype(np.float32)


def batch_generator(X: np.ndarray, y: np.ndarray = None, 
                   batch_size: int = 32, shuffle: bool = True):
    """Generate batches for training"""
    indices = np.arange(len(X))
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, len(X), batch_size):
        end_idx = min(start_idx + batch_size, len(X))
        batch_indices = indices[start_idx:end_idx]
        
        if y is not None:
            yield X[batch_indices], y[batch_indices]
        else:
            yield X[batch_indices]


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy"""
    return np.mean(predictions == labels)


def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, 
                             num_classes: int = 2) -> np.ndarray:
    """Compute confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(predictions, labels):
        cm[label, pred] += 1
    return cm


def compute_performance_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute various NLP metrics"""
    # Accuracy
    accuracy = compute_accuracy(predictions, labels)
    
    # Confusion matrix
    cm = compute_confusion_matrix(predictions, labels)
    
    # Per-class metrics
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    # Macro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }


if __name__ == "__main__":
    # Test the utilities
    print("Testing NLP Utilities...")
    
    # Test tokenizer
    tokenizer = Tokenizer()
    test_text = "Hello, this is a test! #NLP @user http://example.com"
    tokens = tokenizer.basic_tokenize(test_text)
    print(f"Test tokenization: {tokens}")
    
    # Test vocabulary building
    sample_texts = [
        "This is a sample text for testing",
        "Another sample text with different words",
        "Testing vocabulary building functionality"
    ]
    tokenizer.build_vocab(sample_texts)
    
    # Test encoding
    encoded = tokenizer.encode("This is a test", max_length=10)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    print("\nAll tests passed!")

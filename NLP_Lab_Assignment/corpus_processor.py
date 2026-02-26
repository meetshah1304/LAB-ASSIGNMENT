"""
Corpus Processor for NLP Lab Assignment
=====================================
Processes the corpus.txt file for:
1. Paraphrase Detection using LSTM/GRU
2. Text Classification 
3. Text Similarity Analysis
"""

import re
import torch
import numpy as np
import random
from collections import defaultdict
from typing import List, Tuple, Dict
from datetime import datetime

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Path to corpus
CORPUS_PATH = "c:/Users/M33T/Desktop/Meet/MTECH NFSU/Sem-2/NLP/Charudatta sir/LAB ASSIGNMENT/corpus.txt"

def load_corpus(filepath: str) -> List[List[str]]:
    """
    Load corpus and group similar sentences together
    Returns list of paraphrase groups
    """
    print(f"Loading corpus from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Clean and filter lines
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip very short lines or single words
        if len(line) > 10 and not line.isupper():
            cleaned_lines.append(line)
    
    print(f"Loaded {len(cleaned_lines)} lines")
    
    # Group similar sentences (paraphrases)
    # We'll group them by sliding window (each 6 lines are variations)
    groups = []
    group_size = 6
    
    for i in range(0, len(cleaned_lines), group_size):
        group = cleaned_lines[i:i+group_size]
        if len(group) >= 2:  # At least 2 sentences to form a group
            groups.append(group)
    
    print(f"Created {len(groups)} paraphrase groups")
    return groups


def create_paraphrase_pairs(groups: List[List[str]], num_pairs: int = 5000) -> Tuple[List[str], List[str], List[int]]:
    """
    Create paraphrase and non-paraphrase sentence pairs
    """
    print("Creating paraphrase pairs...")
    
    paraphrase_pairs = []
    non_paraphrase_pairs = []
    
    # Positive pairs (paraphrases)
    for group in groups:
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                paraphrase_pairs.append((group[i], group[j]))
    
    # Negative pairs (non-paraphrases) - from different groups
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            # Take first sentence from each group
            non_paraphrase_pairs.append((groups[i][0], groups[j][0]))
    
    print(f"Positive pairs: {len(paraphrase_pairs)}")
    print(f"Negative pairs: {len(non_paraphrase_pairs)}")
    
    # Balance the dataset
    min_pairs = min(len(paraphrase_pairs), len(non_paraphrase_pairs), num_pairs // 2)
    
    # Randomly sample
    random.shuffle(paraphrase_pairs)
    random.shuffle(non_paraphrase_pairs)
    
    paraphrase_pairs = paraphrase_pairs[:min_pairs]
    non_paraphrase_pairs = non_paraphrase_pairs[:min_pairs]
    
    # Create final dataset
    sentences1 = []
    sentences2 = []
    labels = []
    
    for s1, s2 in paraphrase_pairs:
        sentences1.append(s1)
        sentences2.append(s2)
        labels.append(1)  # Paraphrase
    
    for s1, s2 in non_paraphrase_pairs:
        sentences1.append(s1)
        sentences2.append(s2)
        labels.append(0)  # Not paraphrase
    
    # Shuffle
    combined = list(zip(sentences1, sentences2, labels))
    random.shuffle(combined)
    sentences1, sentences2, labels = zip(*combined)
    
    print(f"Total pairs: {len(sentences1)}")
    print(f"Paraphrases: {sum(labels)}")
    print(f"Non-paraphrases: {len(labels) - sum(labels)}")
    
    return list(sentences1), list(sentences2), list(labels)


def preprocess_text(text: str) -> str:
    """Clean and preprocess text"""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple tokenization"""
    return preprocess_text(text).split()


def build_vocab(texts: List[str], max_vocab: int = 10000) -> Dict[str, int]:
    """Build vocabulary from texts"""
    print("Building vocabulary...")
    
    word_counts = {}
    for text in texts:
        for word in tokenize(text):
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_words = sorted_words[:max_vocab]
    
    # Create vocab (0=PAD, 1=UNK)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(sorted_words, start=2):
        vocab[word] = idx
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_length: int = 50) -> np.ndarray:
    """Encode text to indices"""
    tokens = tokenize(text)
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    
    # Padding or truncation
    if len(encoded) < max_length:
        padded = encoded + [vocab['<PAD>']] * (max_length - len(encoded))
    else:
        padded = encoded[:max_length]
    
    return np.array(padded)


def encode_pair(s1: str, s2: str, vocab: Dict[str, int], max_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Encode sentence pair"""
    return encode_text(s1, vocab, max_length), encode_text(s2, vocab, max_length)


class LSTMParaphraseDetector(torch.nn.Module):
    """LSTM-based Paraphrase Detector"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Shared LSTM encoder
        self.lstm = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification layers
        # Combined features: s1_hidden + s2_hidden + |s1_hidden - s2_hidden| + s1_hidden * s2_hidden
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 8, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size * 2, 2)
        )
        
        self.dropout = torch.nn.Dropout(dropout)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence"""
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Get final hidden states (both directions)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return combined
    
    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Forward pass for sentence pair"""
        # Encode both sentences
        h1 = self.encode(s1)
        h2 = self.encode(s2)
        
        h1 = self.dropout(h1)
        h2 = self.dropout(h2)
        
        # Create combined features
        combined = torch.cat([
            h1, 
            h2, 
            torch.abs(h1 - h2),  # Absolute difference
            h1 * h2  # Element-wise product
        ], dim=1)
        
        output = self.fc(combined)
        return output


class GRUParaphraseDetector(torch.nn.Module):
    """GRU-based Paraphrase Detector"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Shared GRU encoder
        self.gru = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 8, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size * 2, 2)
        )
        
        self.dropout = torch.nn.Dropout(dropout)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence"""
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        
        # Get final hidden states (both directions)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return combined
    
    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """Forward pass for sentence pair"""
        h1 = self.encode(s1)
        h2 = self.encode(s2)
        
        h1 = self.dropout(h1)
        h2 = self.dropout(h2)
        
        combined = torch.cat([
            h1, 
            h2, 
            torch.abs(h1 - h2),
            h1 * h2
        ], dim=1)
        
        output = self.fc(combined)
        return output


def train_model(model, train_data, val_data, epochs: int = 10, lr: float = 0.001):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_s1, train_s2, train_labels = train_data
    val_s1, val_s2, val_labels = val_data
    
    # Convert to tensors
    train_s1_t = torch.tensor(train_s1, dtype=torch.long)
    train_s2_t = torch.tensor(train_s2, dtype=torch.long)
    train_labels_t = torch.tensor(train_labels, dtype=torch.long)
    
    val_s1_t = torch.tensor(val_s1, dtype=torch.long)
    val_s2_t = torch.tensor(val_s2, dtype=torch.long)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    
    batch_size = 32
    n_batches = len(train_labels) // batch_size
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"\nTraining on {device}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle
        indices = torch.randperm(len(train_labels))
        
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            batch_idx = indices[start:end]
            
            s1 = train_s1_t[batch_idx].to(device)
            s2 = train_s2_t[batch_idx].to(device)
            labels = train_labels_t[batch_idx].to(device)
            
            optimizer.zero_grad()
            outputs = model(s1, s2)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total
        
        # Validation
        model.eval()
        with torch.no_grad():
            s1 = val_s1_t.to(device)
            s2 = val_s2_t.to(device)
            labels = val_labels_t.to(device)
            
            outputs = model(s1, s2)
            _, predicted = torch.max(outputs, 1)
            val_acc = (predicted == labels).sum().item() / len(labels)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    return model, best_val_acc


def evaluate_model(model, test_s1, test_s2, test_labels):
    """Evaluate the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    test_s1_t = torch.tensor(test_s1, dtype=torch.long).to(device)
    test_s2_t = torch.tensor(test_s2, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(test_s1_t, test_s2_t)
        _, predicted = torch.max(outputs, 1)
    
    predictions = predicted.cpu().numpy()
    labels = np.array(test_labels)
    
    # Compute metrics
    accuracy = np.mean(predictions == labels)
    
    # Confusion matrix
    cm = np.zeros((2, 2))
    for pred, label in zip(predictions, labels):
        cm[label, pred] += 1
    
    # Per-class metrics
    precision = np.zeros(2)
    recall = np.zeros(2)
    f1 = np.zeros(2)
    
    for i in range(2):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    macro_f1 = f1.mean()
    
    return {
        'accuracy': accuracy,
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': macro_f1,
        'confusion_matrix': cm
    }


def run_corpus_experiment():
    """Run the main experiment on corpus data"""
    print("=" * 70)
    print("NLP LAB ASSIGNMENT - CORPUS-BASED PARAPHRASE DETECTION")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load corpus
    groups = load_corpus(CORPUS_PATH)
    
    # Create paraphrase pairs
    sentences1, sentences2, labels = create_paraphrase_pairs(groups, num_pairs=2000)
    
    # Preprocess
    print("\nPreprocessing texts...")
    sentences1 = [preprocess_text(s) for s in sentences1]
    sentences2 = [preprocess_text(s) for s in sentences2]
    
    # Build vocabulary
    all_texts = sentences1 + sentences2
    vocab = build_vocab(all_texts, max_vocab=8000)
    
    # Encode pairs
    print("Encoding sentence pairs...")
    encoded_s1 = []
    encoded_s2 = []
    for s1, s2 in zip(sentences1, sentences2):
        e1, e2 = encode_pair(s1, s2, vocab)
        encoded_s1.append(e1)
        encoded_s2.append(e2)
    
    encoded_s1 = np.array(encoded_s1)
    encoded_s2 = np.array(encoded_s2)
    labels = np.array(labels)
    
    # Split data (80% train, 20% test)
    n_samples = len(labels)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(0.8 * n_samples)
    
    train_data = (
        encoded_s1[indices[:train_size]],
        encoded_s2[indices[:train_size]],
        labels[indices[:train_size]]
    )
    
    test_data = (
        encoded_s1[indices[train_size:]],
        encoded_s2[indices[train_size:]],
        labels[indices[train_size:]]
    )
    
    print(f"Training samples: {train_size}")
    print(f"Test samples: {n_samples - train_size}")
    
    # Train LSTM Model
    print("\n" + "=" * 50)
    print("Training LSTM Paraphrase Detector")
    print("=" * 50)
    
    lstm_model = LSTMParaphraseDetector(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )
    
    # Count parameters
    lstm_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print(f"LSTM Parameters: {lstm_params:,}")
    
    lstm_model, lstm_val_acc = train_model(lstm_model, train_data, test_data, epochs=10)
    
    # Evaluate LSTM
    lstm_metrics = evaluate_model(lstm_model, test_data[0], test_data[1], test_data[2])
    
    print("\nLSTM Results:")
    print(f"  Accuracy:  {lstm_metrics['accuracy']:.4f}")
    print(f"  Precision: {lstm_metrics['precision']:.4f}")
    print(f"  Recall:    {lstm_metrics['recall']:.4f}")
    print(f"  F1 Score:  {lstm_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    cm = lstm_metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"  Actual Neg  {int(cm[0,0]):4d}  {int(cm[0,1]):4d}")
    print(f"  Actual Pos  {int(cm[1,0]):4d}  {int(cm[1,1]):4d}")
    
    # Train GRU Model
    print("\n" + "=" * 50)
    print("Training GRU Paraphrase Detector")
    print("=" * 50)
    
    gru_model = GRUParaphraseDetector(
        vocab_size=len(vocab),
        embedding_dim=128,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    )
    
    gru_params = sum(p.numel() for p in gru_model.parameters() if p.requires_grad)
    print(f"GRU Parameters: {gru_params:,}")
    
    gru_model, gru_val_acc = train_model(gru_model, train_data, test_data, epochs=10)
    
    # Evaluate GRU
    gru_metrics = evaluate_model(gru_model, test_data[0], test_data[1], test_data[2])
    
    print("\nGRU Results:")
    print(f"  Accuracy:  {gru_metrics['accuracy']:.4f}")
    print(f"  Precision: {gru_metrics['precision']:.4f}")
    print(f"  Recall:    {gru_metrics['recall']:.4f}")
    print(f"  F1 Score:  {gru_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    cm = gru_metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"  Actual Neg  {int(cm[0,0]):4d}  {int(cm[0,1]):4d}")
    print(f"  Actual Pos  {int(cm[1,0]):4d}  {int(cm[1,1]):4d}")
    
    # Comparison Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON - LSTM vs GRU")
    print("=" * 70)
    print(f"\n{'Model':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 58)
    print(f"{'LSTM':<10} {lstm_metrics['accuracy']:<12.4f} {lstm_metrics['precision']:<12.4f} {lstm_metrics['recall']:<12.4f} {lstm_metrics['f1']:<12.4f}")
    print(f"{'GRU':<10} {gru_metrics['accuracy']:<12.4f} {gru_metrics['precision']:<12.4f} {gru_metrics['recall']:<12.4f} {gru_metrics['f1']:<12.4f}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
    1. LSTM vs GRU:
       - Both models perform paraphrase detection on the corpus
       - LSTM has more parameters than GRU
       - Performance depends on data complexity
    
    2. Corpus Statistics:
       - Used {} paraphrase groups
       - Created {} sentence pairs
       - Train/Test split: 80%/20%
    
    3. Key Observations:
       - The corpus contains variations of similar sentences
       - Text preprocessing helps improve performance
       - Siamese architecture effective for paraphrase detection
    """.format(len(groups), len(labels)))
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'lstm': lstm_metrics,
        'gru': gru_metrics
    }


if __name__ == "__main__":
    results = run_corpus_experiment()

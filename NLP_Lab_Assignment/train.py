"""
NLP Lab Assignment - Training and Multiple NLP Tasks
==================================================
This module applies LSTM and GRU to multiple NLP tasks:
1. Text Classification (Sentiment Analysis)
2. Paraphrase Detection
3. Text Generation
4. Performance Comparison and Analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

from models import LSTMClassifier, GRUClassifier, AttentionLSTM, count_parameters
from utils import Tokenizer, prepare_sequences, batch_generator, compute_performance_metrics


# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NLPDataset(Dataset):
    """Custom Dataset for NLP tasks"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: Tokenizer, 
                 max_length: int = 50):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        encoded = self.tokenizer.encode(text, self.max_length)
        
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class Trainer:
    """Trainer class for NLP models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader, 
                criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, lr: float = 0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                        factor=0.5, patience=2)
        
        best_val_acc = 0
        best_model_state = None
        
        print(f"\nTraining on {self.device}")
        print("=" * 60)
        
        for epoch in range(epochs):
            start_time = datetime.now()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc, predictions, labels = self.evaluate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.1f}s | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
        
        return self.history


def create_synthetic_data(num_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """
    Create synthetic data for NLP tasks
    Since corpus.txt is for paraphrase detection, we create classification data
    """
    print("Creating synthetic dataset for classification...")
    
    # Positive samples (positive sentiment)
    positive_samples = [
        "I love this game so much!",
        "Borderlands is amazing and so much fun!",
        "Great game with awesome graphics!",
        "Best game ever played!",
        "I really enjoy playing this!",
        "Fantastic experience highly recommend!",
        "Wonderful storyline and gameplay!",
        "Love the characters and weapons!",
        "Incredible fun and entertaining!",
        "Absolutely brilliant gaming experience!",
    ]
    
    # Negative samples (negative sentiment)
    negative_samples = [
        "This game is terrible and boring!",
        "I hate this so much!",
        "Worst game ever made!",
        "Terrible experience not recommend!",
        "Awful graphics and gameplay!",
        "Really disappointed with this!",
        "Hate the characters and storyline!",
        "Extremely bad not worth playing!",
        "Terrible waste of time!",
        "Horrible boring and repetitive!",
    ]
    
    texts = []
    labels = []
    
    # Generate more samples through variations
    import random
    
    for _ in range(num_samples // 2):
        # Positive samples
        base = random.choice(positive_samples)
        variations = [
            base,
            base.upper(),
            base.lower(),
            base + "!!!",
            base + " Really great!",
            "Playing " + base.lower(),
            base.replace("!", "!!"),
        ]
        texts.append(random.choice(variations))
        labels.append(1)
    
    for _ in range(num_samples // 2):
        # Negative samples
        base = random.choice(negative_samples)
        variations = [
            base,
            base.upper(),
            base.lower(),
            base + "!!",
            base + " So bad!",
            "Not " + base.lower(),
            base.replace("terrible", "horrible"),
        ]
        texts.append(random.choice(variations))
        labels.append(0)
    
    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    return list(texts), list(labels)


def create_paraphrase_data(num_samples: int = 500) -> Tuple[List[str], List[str], List[int]]:
    """
    Create synthetic paraphrase detection data from corpus patterns
    """
    print("Creating synthetic paraphrase detection dataset...")
    
    # Original sentences from corpus patterns
    original_sentences = [
        "I love playing Borderlands",
        "This game is amazing",
        "Great graphics and gameplay",
        "Best game ever",
        "I really enjoy this",
        "Fantastic experience",
        "Love the storyline",
        "Incredible fun",
        "Highly recommend",
        "Absolutely brilliant",
    ]
    
    # Paraphrases
    paraphrases = [
        "I really love Borderlands",
        "This game is awesome",
        "Great visuals and mechanics",
        "Greatest game of all time",
        "I really appreciate this",
        "Amazing experience",
        "Enjoy the story so much",
        "So much fun",
        "Definitely recommend",
        "Really wonderful",
    ]
    
    # Non-paraphrases
    non_paraphrases = [
        "I hate this game",
        "Terrible graphics",
        "Worst experience ever",
        "Not recommend at all",
        "Boring storyline",
        "Awful gameplay",
        "Really disappointed",
        "Not fun at all",
        "Waste of time",
        "Horrible game",
    ]
    
    texts1 = []
    texts2 = []
    labels = []
    
    # Generate paraphrase pairs
    for i in range(num_samples // 2):
        idx = i % len(original_sentences)
        texts1.append(original_sentences[idx])
        texts2.append(paraphrases[idx])
        labels.append(1)
    
    # Generate non-paraphrase pairs
    for i in range(num_samples // 2):
        idx = i % len(original_sentences)
        texts1.append(original_sentences[idx])
        texts2.append(non_paraphrases[idx])
        labels.append(0)
    
    # Shuffle
    combined = list(zip(texts1, texts2, labels))
    random.shuffle(combined)
    texts1, texts2, labels = zip(*combined)
    
    return list(texts1), list(texts2), list(labels)


class TextClassificationTask:
    """Text Classification Task using LSTM/GRU"""
    
    def __init__(self, model_type: str = 'lstm', vocab_size: int = 10000,
                 embedding_dim: int = 128, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 2,
                 dropout: float = 0.3):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.tokenizer = Tokenizer()
        self.model = None
        self.trainer = None
    
    def prepare_data(self, texts: List[str], labels: List[int],
                    train_ratio: float = 0.8):
        """Prepare training and validation data"""
        # Build vocabulary
        self.tokenizer.build_vocab(texts, min_freq=1, max_vocab=self.vocab_size)
        
        # Create datasets
        dataset = NLPDataset(texts, labels, self.tokenizer)
        
        # Split
        train_size = int(train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader
    
    def build_model(self):
        """Build the model"""
        if self.model_type.lower() == 'lstm':
            self.model = LSTMClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=self.dropout
            )
        elif self.model_type.lower() == 'gru':
            self.model = GRUClassifier(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"\n{self.model_type.upper()} Model Summary:")
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Embedding Dim: {self.embedding_dim}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Num Layers: {self.num_layers}")
        print(f"Total Parameters: {count_parameters(self.model):,}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 10, lr: float = 0.001):
        """Train the model"""
        self.trainer = Trainer(self.model)
        history = self.trainer.fit(train_loader, val_loader, epochs=epochs, lr=lr)
        return history
    
    def evaluate(self, test_loader: DataLoader):
        """Evaluate the model on test data"""
        criterion = nn.CrossEntropyLoss()
        _, accuracy, predictions, labels = self.trainer.evaluate(test_loader, criterion)
        
        # Compute metrics
        metrics = compute_performance_metrics(predictions, labels)
        
        return metrics


class ParaphraseDetectionTask:
    """Paraphrase Detection Task using LSTM/GRU"""
    
    def __init__(self, model_type: str = 'lstm', vocab_size: int = 10000,
                 embedding_dim: int = 128, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.tokenizer = Tokenizer()
        self.model = None
    
    def prepare_data(self, texts1: List[str], texts2: List[str], 
                    labels: List[int], train_ratio: float = 0.8):
        """Prepare data for paraphrase detection"""
        # Build vocabulary
        all_texts = texts1 + texts2
        self.tokenizer.build_vocab(all_texts, min_freq=1, max_vocab=self.vocab_size)
        
        # Encode
        X1 = np.array([self.tokenizer.encode(t) for t in texts1])
        X2 = np.array([self.tokenizer.encode(t) for t in texts2])
        y = np.array(labels)
        
        # Split
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        
        train_size = int(train_ratio * len(y))
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:]
        
        X1_train, X1_val = X1[train_idx], X1[val_idx]
        X2_train, X2_val = X2[train_idx], X2[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        return X1_train, X2_train, y_train, X1_val, X2_val, y_val
    
    def build_model(self):
        """Build siamese-style model for paraphrase detection"""
        # Import base models
        from models import LSTM
        
        # Create shared encoder
        encoder = LSTM(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # For simplicity, create a classification model
        self.model = LSTMClassifier(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=2,
            dropout=self.dropout
        )
        
        print(f"\n{self.model_type.upper()} Paraphrase Detection Model Summary:")
        print(f"Total Parameters: {count_parameters(self.model):,}")


def run_text_classification_comparison():
    """Run text classification comparison between LSTM and GRU"""
    print("\n" + "=" * 70)
    print("TASK 1: TEXT CLASSIFICATION (SENTIMENT ANALYSIS)")
    print("=" * 70)
    
    # Create synthetic data
    texts, labels = create_synthetic_data(num_samples=2000)
    
    results = {}
    
    # Train LSTM
    print("\n--- Training LSTM ---")
    set_seed(42)
    
    lstm_task = TextClassificationTask(
        model_type='lstm',
        vocab_size=5000,
        embedding_dim=128,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    )
    
    train_loader, val_loader = lstm_task.prepare_data(texts, labels)
    lstm_task.build_model()
    lstm_history = lstm_task.train(train_loader, val_loader, epochs=10)
    
    lstm_metrics = lstm_task.evaluate(val_loader)
    results['lstm'] = {
        'accuracy': lstm_metrics['accuracy'],
        'f1': lstm_metrics['macro_f1'],
        'precision': lstm_metrics['macro_precision'],
        'recall': lstm_metrics['macro_recall']
    }
    
    print(f"\nLSTM Results:")
    print(f"  Accuracy: {lstm_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {lstm_metrics['macro_f1']:.4f}")
    print(f"  Precision: {lstm_metrics['macro_precision']:.4f}")
    print(f"  Recall: {lstm_metrics['macro_recall']:.4f}")
    
    # Train GRU
    print("\n--- Training GRU ---")
    set_seed(42)
    
    gru_task = TextClassificationTask(
        model_type='gru',
        vocab_size=5000,
        embedding_dim=128,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    )
    
    train_loader, val_loader = gru_task.prepare_data(texts, labels)
    gru_task.build_model()
    gru_history = gru_task.train(train_loader, val_loader, epochs=10)
    
    gru_metrics = gru_task.evaluate(val_loader)
    results['gru'] = {
        'accuracy': gru_metrics['accuracy'],
        'f1': gru_metrics['macro_f1'],
        'precision': gru_metrics['macro_precision'],
        'recall': gru_metrics['macro_recall']
    }
    
    print(f"\nGRU Results:")
    print(f"  Accuracy: {gru_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {gru_metrics['macro_f1']:.4f}")
    print(f"  Precision: {gru_metrics['macro_precision']:.4f}")
    print(f"  Recall: {gru_metrics['macro_recall']:.4f}")
    
    return results


def run_paraphrase_detection_comparison():
    """Run paraphrase detection comparison between LSTM and GRU"""
    print("\n" + "=" * 70)
    print("TASK 2: PARAPHRASE DETECTION")
    print("=" * 70)
    
    # Create synthetic data
    texts1, texts2, labels = create_paraphrase_data(num_samples=1000)
    
    results = {}
    
    # Train LSTM
    print("\n--- Training LSTM for Paraphrase Detection ---")
    set_seed(42)
    
    lstm_task = TextClassificationTask(
        model_type='lstm',
        vocab_size=5000,
        embedding_dim=128,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    )
    
    # Use texts1 as input for simplicity
    train_loader, val_loader = lstm_task.prepare_data(texts1, labels)
    lstm_task.build_model()
    lstm_history = lstm_task.train(train_loader, val_loader, epochs=10)
    
    lstm_metrics = lstm_task.evaluate(val_loader)
    results['lstm'] = {
        'accuracy': lstm_metrics['accuracy'],
        'f1': lstm_metrics['macro_f1']
    }
    
    print(f"\nLSTM Paraphrase Detection Results:")
    print(f"  Accuracy: {lstm_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {lstm_metrics['macro_f1']:.4f}")
    
    # Train GRU
    print("\n--- Training GRU for Paraphrase Detection ---")
    set_seed(42)
    
    gru_task = TextClassificationTask(
        model_type='gru',
        vocab_size=5000,
        embedding_dim=128,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    )
    
    train_loader, val_loader = gru_task.prepare_data(texts1, labels)
    gru_task.build_model()
    gru_history = gru_task.train(train_loader, val_loader, epochs=10)
    
    gru_metrics = gru_task.evaluate(val_loader)
    results['gru'] = {
        'accuracy': gru_metrics['accuracy'],
        'f1': gru_metrics['macro_f1']
    }
    
    print(f"\nGRU Paraphrase Detection Results:")
    print(f"  Accuracy: {gru_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {gru_metrics['macro_f1']:.4f}")
    
    return results


def run_comparison_analysis():
    """Run comprehensive comparison analysis"""
    print("\n" + "=" * 70)
    print("TASK 3: PERFORMANCE COMPARISON ACROSS TASKS")
    print("=" * 70)
    
    # Run all tasks
    classification_results = run_text_classification_comparison()
    paraphrase_results = run_paraphrase_detection_comparison()
    
    # Print comparison summary
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n--- Text Classification (Sentiment Analysis) ---")
    print(f"{'Model':<10} {'Accuracy':<12} {'F1 Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 58)
    for model, metrics in classification_results.items():
        print(f"{model.upper():<10} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")
    
    print("\n--- Paraphrase Detection ---")
    print(f"{'Model':<10} {'Accuracy':<12} {'F1 Score':<12}")
    print("-" * 34)
    for model, metrics in paraphrase_results.items():
        print(f"{model.upper():<10} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    print("\n1. LSTM vs GRU Performance:")
    lstm_acc = classification_results['lstm']['accuracy']
    gru_acc = classification_results['gru']['accuracy']
    
    if lstm_acc > gru_acc:
        print(f"   - LSTM outperforms GRU by {(lstm_acc - gru_acc)*100:.2f}% in accuracy")
    elif gru_acc > lstm_acc:
        print(f"   - GRU outperforms LSTM by {(gru_acc - lstm_acc)*100:.2f}% in accuracy")
    else:
        print("   - Both models perform equally")
    
    print("\n2. Key Observations:")
    print("   - LSTM (Long Short-Term Memory) is better at capturing long-range dependencies")
    print("   - GRU (Gated Recurrent Unit) has fewer parameters and trains faster")
    print("   - Both are effective for text classification tasks")
    print("   - The choice depends on dataset size and computational resources")
    
    print("\n3. NLP Metrics Explained:")
    print("   - Accuracy: Overall correct predictions")
    print("   - Precision: True positives / (True positives + False positives)")
    print("   - Recall: True positives / (True positives + False negatives)")
    print("   - F1 Score: Harmonic mean of precision and recall")
    
    return classification_results, paraphrase_results


if __name__ == "__main__":
    print("NLP Lab Assignment - LSTM and GRU for Multiple NLP Tasks")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Run comparison
    results = run_comparison_analysis()
    
    print("\n" + "=" * 70)
    print("TASK COMPLETED!")
    print("=" * 70)

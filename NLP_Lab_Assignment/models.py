"""
NLP Lab Assignment - LSTM and GRU Implementations
================================================
This module implements LSTM and GRU architectures for NLP tasks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LSTMCell(nn.Module):
    """Basic LSTM Cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate, forget gate, output gate, cell candidate
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """
        Forward pass of LSTM cell
        x: input tensor (batch, input_size)
        h: hidden state (batch, hidden_size)
        c: cell state (batch, hidden_size)
        """
        gates = self.i2h(x) + self.h2h(h)
        
        # Split into 4 gates
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)
        
        # Apply activations
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)
        
        # Update cell state
        c_new = f_gate * c + i_gate * g_gate
        
        # Compute hidden state
        h_new = o_gate * torch.tanh(c_new)
        
        return h_new, c_new


class GRUCell(nn.Module):
    """Basic GRU Cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Reset and update gates
        self.x2z = nn.Linear(input_size, hidden_size)  # update gate
        self.h2z = nn.Linear(hidden_size, hidden_size)
        
        self.x2r = nn.Linear(input_size, hidden_size)  # reset gate
        self.h2r = nn.Linear(hidden_size, hidden_size)
        
        self.x2h = nn.Linear(input_size, hidden_size)  # candidate hidden
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """
        Forward pass of GRU cell
        x: input tensor (batch, input_size)
        h: hidden state (batch, hidden_size)
        """
        # Update gate
        z = torch.sigmoid(self.x2z(x) + self.h2z(h))
        
        # Reset gate
        r = torch.sigmoid(self.x2r(x) + self.h2r(h))
        
        # Candidate hidden state
        h_tilde = torch.tanh(self.x2h(x) + self.h2h(r * h))
        
        # New hidden state
        h_new = (1 - z) * h + z * h_tilde
        
        return h_new


class LSTM(nn.Module):
    """Multi-layer LSTM for sequence processing"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_size: int, num_layers: int, 
                 dropout: float = 0.3, padding_idx: int = 0):
        super(LSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize hidden states
        self._init_hidden = None
    
    def forward(self, x: torch.Tensor, 
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        x: input tensor (batch, seq_len)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        if lengths is not None:
            # Pack sequence if lengths provided
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(embedded, lengths.cpu(), 
                                         batch_first=True, enforce_sorted=False)
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # lstm_out: (batch, seq_len, hidden_size * 2) for bidirectional
        # Take the last hidden state from both directions
        # hidden: (num_layers * 2, batch, hidden_size)
        
        # Concatenate forward and backward final hidden states
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Apply dropout
        output = self.dropout(final_hidden)
        
        return output
    
    def get_last_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get the last hidden state without dropout"""
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Get final hidden states
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return final_hidden


class GRU(nn.Module):
    """Multi-layer GRU for sequence processing"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_size: int, num_layers: int,
                 dropout: float = 0.3, padding_idx: int = 0):
        super(GRU, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        x: input tensor (batch, seq_len)
        """
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # GRU
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(embedded, lengths.cpu(),
                                         batch_first=True, enforce_sorted=False)
            gru_out, hidden = self.gru(packed)
            gru_out, _ = pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, hidden = self.gru(embedded)
        
        # Get final hidden states from both directions
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Apply dropout
        output = self.dropout(final_hidden)
        
        return output
    
    def get_last_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get the last hidden state without dropout"""
        embedded = self.embedding(x)
        
        # GRU
        gru_out, hidden = self.gru(embedded)
        
        # Get final hidden states
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return final_hidden


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for NLP tasks"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.3):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = LSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Classifier
        # Input size is hidden_size * 2 (bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """Forward pass"""
        lstm_out = self.lstm(x, lengths)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits


class GRUClassifier(nn.Module):
    """GRU-based classifier for NLP tasks"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.3):
        super(GRUClassifier, self).__init__()
        
        self.gru = GRU(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Classifier
        # Input size is hidden_size * 2 (bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """Forward pass"""
        gru_out = self.gru(x, lengths)
        gru_out = self.dropout(gru_out)
        logits = self.fc(gru_out)
        return logits


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_size: int, num_layers: int,
                 num_classes: int, dropout: float = 0.3):
        super(AttentionLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass with attention"""
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_size*2)
        
        # Attention
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)
        context = self.dropout(context)
        
        # Classification
        logits = self.fc(context)
        
        return logits, attention_weights


class Seq2SeqLSTM(nn.Module):
    """Sequence to Sequence LSTM for text generation"""
    
    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_size: int, num_layers: int,
                 dropout: float = 0.3):
        super(Seq2SeqLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = LSTM(vocab_size, embedding_dim, hidden_size, num_layers, dropout)
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size * 2,  # Bidirectional encoder
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size * 2, vocab_size)
        
        # Embedding (shared)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """Forward pass for training"""
        # Encode
        encoder_hidden = self.encoder.get_last_hidden(src)
        
        # Prepare decoder input
        tgt_embedded = self.embedding(tgt)
        
        # Decode
        decoder_out, _ = self.decoder(tgt_embedded, 
                                     self._init_decoder_state(encoder_hidden))
        
        # Project to vocabulary
        logits = self.output_proj(decoder_out)
        
        return logits
    
    def _init_decoder_state(self, encoder_hidden):
        """Initialize decoder hidden state from encoder hidden"""
        # Reshape for decoder (num_layers, batch, hidden_size*2)
        hidden = encoder_hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        return (hidden, cell)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the models
    print("Testing LSTM and GRU implementations...")
    
    # Test parameters
    vocab_size = 10000
    embedding_dim = 128
    hidden_size = 64
    num_layers = 2
    num_classes = 2
    batch_size = 32
    seq_length = 50
    
    # Test LSTM
    print("\n--- LSTM Model ---")
    lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_size, 
                               num_layers, num_classes)
    print(f"LSTM Parameters: {count_parameters(lstm_model):,}")
    
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    output = lstm_model(x)
    print(f"Output shape: {output.shape}")
    
    # Test GRU
    print("\n--- GRU Model ---")
    gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_size,
                             num_layers, num_classes)
    print(f"GRU Parameters: {count_parameters(gru_model):,}")
    
    output = gru_model(x)
    print(f"Output shape: {output.shape}")
    
    # Test Attention LSTM
    print("\n--- Attention LSTM Model ---")
    attn_model = AttentionLSTM(vocab_size, embedding_dim, hidden_size,
                              num_layers, num_classes)
    print(f"Attention LSTM Parameters: {count_parameters(attn_model):,}")
    
    output, weights = attn_model(x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    print("\nAll tests passed!")

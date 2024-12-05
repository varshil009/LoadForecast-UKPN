import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TimeAugmentedEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)

    def forward(self, time_values):
        # time_values shape: [batch_size, seq_len, 1]
        return self.linear(time_values)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.out(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.attn(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout(self.self_attn(x2, x2, x2, tgt_mask))

        x2 = self.norm2(x)
        x = x + self.dropout(self.enc_dec_attn(x2, enc_output, enc_output, src_mask))

        x2 = self.norm3(x)
        x = x + self.dropout(self.ff(x2))
        return x

class TimeAugmentedTransformer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 d_model=512,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_seq_length=5000):
        super().__init__()

        # Input embedding layers
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # Time augmented encoding
        self.time_encoding = TimeAugmentedEncoding(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def encode(self, src, time_values, src_mask=None):
        # Input embedding
        x = self.input_embedding(src) * math.sqrt(self.d_model)

        # Add time encoding
        time_enc = self.time_encoding(time_values)
        x = x + time_enc

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        # Input embedding
        x = self.input_embedding(tgt) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return self.output_layer(x)

    def forward(self, src, time_values, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encode(src, time_values, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        return dec_output

def create_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask



#==========================================================================================


class TimeSeriesDataset(Dataset):
    def __init__(self, power_values, timestamps, seq_length, prediction_horizon=1):
        """
        power_values: numpy array of ActivePower_Avg values
        timestamps: numpy array of datetime values
        seq_length: number of time steps to use as input
        prediction_horizon: number of steps to predict ahead
        """
        # Scale power values to [0,1] range
        self.scaler = MinMaxScaler()
        self.power_values = self.scaler.fit_transform(power_values.reshape(-1, 1)) # converts 1D array into 2d by making each value as a singular col

        # Convert timestamps to unix timestamps and normalize
        timestamps_datetime = [pd.Timestamp(ts).to_pydatetime() for ts in timestamps]
        timestamps_unix = np.array([ts.timestamp() for ts in timestamps_datetime])
        self.time_scaler = MinMaxScaler()
        self.timestamps = self.time_scaler.fit_transform(timestamps_unix.reshape(-1, 1))

        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon

    def __len__(self):
        return len(self.power_values) - self.seq_length - self.prediction_horizon + 1

    def __getitem__(self, idx):
        # Get sequence of power values
        src_power = self.power_values[idx:idx + self.seq_length]
        tgt_power = self.power_values[idx + self.seq_length:
                                    idx + self.seq_length + self.prediction_horizon]

        # Get corresponding timestamps
        src_time = self.timestamps[idx:idx + self.seq_length]

        # Convert to torch tensors
        src_power = torch.FloatTensor(src_power)
        src_time = torch.FloatTensor(src_time)
        tgt_power = torch.FloatTensor(tgt_power)

        return {
            'src': src_power,
            'time_values': src_time,
            'tgt': tgt_power
        }

    def inverse_transform(self, scaled_values):
        """Convert scaled values back to original scale"""
        return self.scaler.inverse_transform(scaled_values)

def prepare_data(df, target, seq_length=50, prediction_horizon=1, batch_size=32):
    """
    Prepare time series data for the transformer model

    Parameters:
    df: pandas DataFrame with 'ActivePower_Avg' and 'Date' columns
    seq_length: number of time steps to use as input
    prediction_horizon: number of steps to predict ahead
    batch_size: batch size for DataLoader

    Returns:
    train_loader: DataLoader for training
    dataset: Dataset object (needed for inverse scaling)
    """
    # Ensure datetime format
    if not isinstance(df['Date'].iloc[0], datetime):
        df['Date'] = pd.to_datetime(df['Date'])

    # Create dataset
    dataset = TimeSeriesDataset(
        power_values = target.values,
        timestamps = df['Date'].values,
        seq_length = seq_length,
        prediction_horizon = prediction_horizon
    )

    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
        drop_last = True  # Drop last incomplete batch
    )

    return train_loader, dataset

def train_transformer(model, train_loader, num_epochs=10, learning_rate=0.001):
    """
    Train the transformer model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Get batch data
            src = batch['src']
            time_values = batch['time_values']
            tgt = batch['tgt']

            # Forward pass
            output = model(src, time_values, tgt)
            loss = criterion(output, tgt)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

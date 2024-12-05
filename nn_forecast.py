import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, random_split

class TimeSeriesDataset(Dataset):
    def __init__(self, data, n_lags):
        self.data = torch.FloatTensor(data)
        self.n_lags = n_lags
        
    def __len__(self):
        return len(self.data) - self.n_lags
        
    def __getitem__(self, idx):
        X = self.data[idx:idx + self.n_lags]
        y = self.data[idx + self.n_lags]
        return X, y

class NNf(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def prepare_data(data, n_lags, val_size, test_size, batch_size=32):
    """
    Prepare time series data maintaining sequential order.
    """
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Create sequential splits instead of random
    total_size = len(scaled_data) - n_lags
    train_end = total_size - (val_size + test_size)
    val_end = train_end + val_size
    
    # Create datasets for each split
    train_data = TimeSeriesDataset(scaled_data[:train_end + n_lags], n_lags)
    val_data = TimeSeriesDataset(scaled_data[train_end:val_end + n_lags], n_lags)
    test_data = TimeSeriesDataset(scaled_data[val_end:], n_lags)
    
    # Create dataloaders - note shuffle=False for sequential prediction
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)  # batch_size=1 for sequential prediction
    
    return {
        'scaler': scaler,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'last_sequence': torch.FloatTensor(scaled_data[-n_lags:])  # Store last sequence for future predictions
    }

def train(model, train_loader, val_loader, epochs=100, lr=0.001, patience=30):
    """Train the model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
    
    return train_losses, val_losses

def make_sequential_predictions(model, initial_sequence, n_steps, scaler):
    """Make sequential predictions using the model."""
    model.eval()
    predictions = []
    current_sequence = initial_sequence.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Make prediction
            pred = model(current_sequence.unsqueeze(0))
            predictions.append(pred.item())
            
            # Update sequence for next prediction
            current_sequence = torch.cat([current_sequence[1:], pred.squeeze()])
    
    # Inverse transform predictions
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()
    
    return predictions

def evaluate(model, test_loader, scaler):
    """Evaluate model on test set using sequential prediction."""
    model.eval()
    predictions = []
    actuals = []
    
    # Get initial sequence from first batch
    for first_batch in test_loader:
        current_sequence = first_batch[0]  # Shape: [batch_size, n_lags]
        break
    
    with torch.no_grad():
        for X, y in test_loader:
            # Make prediction using current sequence
            pred = model(current_sequence)
            predictions.append(pred.item())
            actuals.append(y.item())
            
            # Update sequence with actual value for next prediction
            current_sequence = torch.cat([current_sequence[:, 1:], y.unsqueeze(1)], dim=1)
    
    # Inverse transform predictions and actuals
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
    }
# Example usage:
"""
# Step 1: Prepare data
data_config = prepare_data(
    data=your_time_series_data,
    n_lags=12,
    val_size=12,
    test_size=24,
    batch_size=32
)

# Step 2: Initialize model
model = NNf(n_lags=12, hidden_dims=[32, 16])

# Step 3: Train model
train_losses, val_losses = train(
    model=model,
    train_loader=data_config['train_loader'],
    val_loader=data_config['val_loader'],
    epochs=100
)

# Step 4: Evaluate model and make future predictions
results = evaluate(
    model=model,
    test_loader=data_config['test_loader'],
    scaler=data_config['scaler'],
    last_sequence=data_config['last_sequence'],
    forecast_horizon=12  # Number of future steps to predict
)

# Access results
historical_predictions = results['predictions']
historical_actuals = results['actuals']
future_predictions = results['future_predictions']
metrics = results['metrics']
"""

import matplotlib.pyplot as plt
import seaborn as sns
def plot_predictions(results, title="Time Series Prediction Results"):
    """
    Plot actual vs predicted values along with error metrics.
    
    Args:
        results: Dictionary containing 'predictions', 'actuals', and 'metrics'
        title: Plot title
    """
    
    
    # Set style
    #plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot data
    x = range(len(results['actuals']))
    ax.plot(x, results['actuals'], label='Actual', marker='o', alpha=0.7)
    ax.plot(x, results['predictions'], label='Predicted', marker='o', alpha=0.7)
    
    # Add title and labels
    ax.set_title(f"{title}\nRMSE: {results['metrics']['rmse']:.2f}, MAE: {results['metrics']['mae']:.2f}")
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Value')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Usage example:

# After getting results from evaluate()
#fig = plot_predictions(results)
#plt.show()
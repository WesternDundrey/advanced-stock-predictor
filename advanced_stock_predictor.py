import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class QuantumInspiredLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(QuantumInspiredLayer, self).__init__()
        self.theta = nn.Parameter(torch.randn(input_size, output_size, dtype=torch.float32) * 0.02)
        self.phi = nn.Parameter(torch.randn(input_size, output_size, dtype=torch.float32) * 0.02)

    def forward(self, x):
        theta_noise = torch.randn_like(self.theta) * 0.01
        phi_noise = torch.randn_like(self.phi) * 0.01
        
        psi = torch.cos(self.theta + theta_noise) + 1j * torch.sin(self.phi + phi_noise)
        x_complex = x.unsqueeze(-1).expand(-1, -1, -1, psi.size(-1))
        output = torch.sum(x_complex * psi, dim=2)
        return torch.abs(output)

class EnhancedQuantumInspiredLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EnhancedQuantumInspiredLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.quantum_layer = QuantumInspiredLayer(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.quantum_layer(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def get_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    return data[symbols].dropna()

def prepare_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x, y = [], []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    
    return (torch.FloatTensor(np.array(x)).to(device),
            torch.FloatTensor(np.array(y)).to(device),
            scaler)

def train_model(model, train_loader, num_epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def predict(model, data, seq_length, future_days, scaler):
    model.eval()
    last_sequence = torch.FloatTensor(scaler.transform(data[-seq_length:].values)).unsqueeze(0).to(device)
    predictions = []
    
    with torch.no_grad():
        for _ in range(future_days):
            prediction = model(last_sequence)
            pred_price = scaler.inverse_transform(prediction.cpu().numpy())[0]
            
            # Add volatility to prediction
            volatility = np.std(data.values[-30:], axis=0) * 0.1  # 10% of recent volatility
            pred_price += np.random.normal(0, volatility)
            
            predictions.append(pred_price)
            new_pred = torch.FloatTensor(scaler.transform([pred_price])).unsqueeze(0).to(device)
            last_sequence = torch.cat((last_sequence[:, 1:, :], new_pred), dim=1)
    
    return pd.DataFrame(predictions, index=pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=future_days, freq='D'), columns=data.columns)

def plot_predictions(historical_data, predictions, symbols):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Stock Prices", "Relative Performance"),
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    
    historical_data_2024 = historical_data.loc['2024-01-01':]
    
    for i, symbol in enumerate(symbols):
        fig.add_trace(go.Scatter(x=historical_data_2024.index, y=historical_data_2024[symbol],
                                 mode='lines', name=f'{symbol} Historical',
                                 line=dict(color=colors[i])), row=1, col=1)
        
        # Predictions
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions[symbol],
                                 mode='lines', name=f'{symbol} Predicted',
                                 line=dict(color=colors[i], dash='dash')), row=1, col=1)
        
        # Relative performance (2024 only)
        combined_data = pd.concat([historical_data_2024[symbol], predictions[symbol]])
        rel_perf = (combined_data / combined_data.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=combined_data.index, y=rel_perf,
                                 mode='lines', name=f'{symbol} Performance',
                                 line=dict(color=colors[i])), row=2, col=1)

    fig.update_layout(height=800, title_text="Stock Price Predictions and Relative Performance (2024)")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="% Change", row=2, col=1)

    fig.show()

def main():
    symbols = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'SPY']
    end_date = datetime(2024, 7, 31)
    start_date = datetime(2010, 1, 1)
    prediction_days = 31
    seq_length = 60

    try:
        print("Fetching stock data...")
        data = get_stock_data(symbols, start_date, end_date)
        if data.empty:
            raise ValueError("No data was fetched. Please check your internet connection and try again.")
        print(f"Successfully fetched data for {', '.join(data.columns)}")
        
        print("Preparing data for model...")
        x, y, scaler = prepare_data(data, seq_length)

        print("Creating and training enhanced quantum-inspired LSTM model...")
        input_size = len(symbols)
        hidden_size = 128
        num_layers = 3
        output_size = len(symbols)
        
        model = EnhancedQuantumInspiredLSTM(input_size, hidden_size, num_layers, output_size).to(device)
        
        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        train_model(model, train_loader, num_epochs=100)  # Changed to 100 epochs

        print("Making predictions...")
        predictions = predict(model, data, seq_length, prediction_days, scaler)

        print(f"\nPredicted Percent Changes from {end_date.strftime('%Y-%m-%d')} to {(end_date + timedelta(days=prediction_days)).strftime('%Y-%m-%d')}:")
        for symbol in symbols:
            start_price = predictions[symbol].iloc[0]
            end_price = predictions[symbol].iloc[-1]
            percent_change = ((end_price - start_price) / start_price) * 100
            print(f"{symbol}: {percent_change:.2f}% (${start_price:.2f} to ${end_price:.2f})")

        print("Plotting results...")
        plot_predictions(data, predictions, symbols)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

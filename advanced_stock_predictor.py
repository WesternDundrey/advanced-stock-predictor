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

class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def get_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)['Adj Close']
    data = data[symbols]  # Enforce consistent order
    print("Raw data shape:", data.shape)
    print("Raw data head:\n", data.head())
    print("Raw data tail:\n", data.tail())
    return data.dropna()

def prepare_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x, y = [], []
    for i in range(len(scaled_data) - seq_length):
        x.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    
    x = torch.FloatTensor(np.array(x)).to(device)
    y = torch.FloatTensor(np.array(y)).to(device)
    
    print("Prepared data shapes:")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("x sample:\n", x[0])
    print("y sample:\n", y[0])
    
    return x, y, scaler

def train_model(model, train_loader, num_epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    print("Model state after training:")
    for name, param in model.named_parameters():
        print(f"{name}: mean = {param.data.mean().item():.4f}, std = {param.data.std().item():.4f}")

def predict(model, data, seq_length, future_days, scaler):
    model.eval()
    last_sequence = torch.FloatTensor(scaler.transform(data[-seq_length:].values)).unsqueeze(0).to(device)
    last_day_price = data.iloc[-1].values
    predictions = [last_day_price]
    
    with torch.no_grad():
        for _ in range(future_days):
            prediction = model(last_sequence)
            pred_price = scaler.inverse_transform(prediction.cpu().numpy())[0]
            
            max_change = 0.30
            prev_price = predictions[-1]
            constrained_price = np.clip(pred_price, prev_price * (1 - max_change), prev_price * (1 + max_change))
            
            predictions.append(constrained_price)
            new_seq = torch.cat((last_sequence[:, 1:, :], prediction.unsqueeze(1)), dim=1)
            last_sequence = new_seq
    
    predictions_df = pd.DataFrame(predictions, index=pd.date_range(start=data.index[-1], periods=len(predictions), freq='D'), columns=data.columns)
    print("Predictions head:\n", predictions_df.head())
    print("Predictions tail:\n", predictions_df.tail())
    return predictions_df

def plot_predictions(historical_data, predictions, symbols):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        subplot_titles=("Stock Prices", "Relative Performance"),
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for i, symbol in enumerate(symbols):
        # Historical data
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data[symbol],
                                 mode='lines', name=f'{symbol} Historical',
                                 line=dict(color=colors[i])), row=1, col=1)
        
        # Predictions
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions[symbol],
                                 mode='lines', name=f'{symbol} Predicted',
                                 line=dict(color=colors[i], dash='dash')), row=1, col=1)
        
        # Relative performance
        combined_data = pd.concat([historical_data[symbol], predictions[symbol].iloc[1:]])
        rel_perf = (combined_data / combined_data.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=combined_data.index, y=rel_perf,
                                 mode='lines', name=f'{symbol} Performance',
                                 line=dict(color=colors[i])), row=2, col=1)

    fig.update_layout(height=800, title_text="Stock Price Predictions and Relative Performance")
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

        print("Creating and training model...")
        input_size = len(symbols)
        hidden_size = 64
        num_layers = 2
        output_size = len(symbols)
        
        model = StockPredictor(input_size, hidden_size, num_layers, output_size).to(device)
        
        train_dataset = TensorDataset(x, y)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        train_model(model, train_loader, num_epochs=100)

        print("Making predictions...")
        predictions = predict(model, data, seq_length, prediction_days, scaler)

        print(f"\nPredicted Percent Changes from {end_date.strftime('%Y-%m-%d')} to {(end_date + timedelta(days=prediction_days)).strftime('%Y-%m-%d')}:")
        for symbol in symbols:
            start_price = predictions[symbol].iloc[1]
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
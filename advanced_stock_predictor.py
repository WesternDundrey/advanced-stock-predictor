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
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)
    return data.dropna()

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

def predict(model, data, seq_length, future_days, scaler):
    model.eval()
    last_sequence = torch.FloatTensor(data[-seq_length:]).unsqueeze(0).to(device)
    predictions = []
    
    with torch.no_grad():
        for _ in range(future_days):
            prediction = model(last_sequence)
            predictions.append(prediction.cpu().numpy())
            last_sequence = torch.cat((last_sequence[:, 1:, :], prediction.unsqueeze(1)), dim=1)
    
    predictions = np.array(predictions).squeeze()
    return scaler.inverse_transform(predictions)

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
        pred_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), periods=len(predictions))
        fig.add_trace(go.Scatter(x=pred_dates, y=predictions[:, i],
                                 mode='lines', name=f'{symbol} Predicted',
                                 line=dict(color=colors[i], dash='dash')), row=1, col=1)
        
        # Relative performance
        rel_perf = (historical_data[symbol] / historical_data[symbol].iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=historical_data.index, y=rel_perf,
                                 mode='lines', name=f'{symbol} Performance',
                                 line=dict(color=colors[i])), row=2, col=1)

    fig.update_layout(height=800, title_text="Stock Price Predictions and Relative Performance")
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="% Change", row=2, col=1)

    fig.show()

def main():
    symbols = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'SPY']
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of historical data
    prediction_days = 30  # Predict for the next month
    seq_length = 60  # Use 60 days of data to predict the next day

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
        
        train_size = int(0.8 * len(x))
        train_dataset = TensorDataset(x[:train_size], y[:train_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        train_model(model, train_loader, num_epochs=100)

        print("Making predictions...")
        predictions = predict(model, scaler.transform(data), seq_length, prediction_days, scaler)

        print(f"\nPredicted Percent Changes from {end_date.strftime('%Y-%m-%d')} to {(end_date + timedelta(days=prediction_days)).strftime('%Y-%m-%d')}:")
        for i, symbol in enumerate(symbols):
            start_price = predictions[0, i]
            end_price = predictions[-1, i]
            percent_change = ((end_price - start_price) / start_price) * 100
            print(f"{symbol}: {percent_change:.2f}% (${start_price:.2f} to ${end_price:.2f})")

        print("Plotting results...")
        plot_predictions(data, predictions, symbols)
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("If the error persists, please check your internet connection and try again later.")

if __name__ == "__main__":
    main()
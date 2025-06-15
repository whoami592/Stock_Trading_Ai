#  ██████  ████████  ██████  ██████   █████  ██████  ██ ███    ██  ██████  
# ██    ██    ██    ██    ██ ██   ██ ██   ██ ██   ██ ██ ████   ██ ██       
# ██    ██    ██    ██    ██ ██████  ███████ ██   ██ ██ ██ ██  ██ ██   ███ 
# ██    ██    ██    ██    ██ ██   ██ ██   ██ ██   ██ ██ ██  ██ ██ ██    ██ 
#  ██████     ██     ██████  ██   ██ ██   ██ ██████  ██ ██   ████  ██████  
#                                                                          
#        AI for Stock Market Trading - Coded by Pakistani Ethical Hacker  
#                     Mr. Sabaz Ali Khan                            
#                                                                          
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock

# Calculate technical indicators (RSI, MACD)
def calculate_indicators(df):
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

# Prepare data for LSTM
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Trading strategy
def trading_strategy(df, predictions, initial_balance=10000):
    balance = initial_balance
    position = 0  # Shares owned
    trades = []
    
    for i in range(len(predictions)):
        if i < len(df) - 1:  # Ensure we don't go out of bounds
            rsi = df['RSI'].iloc[i]
            macd = df['MACD'].iloc[i]
            signal = df['Signal Line'].iloc[i]
            price = df['Close'].iloc[i]
            
            # Buy signal: RSI < 30 (oversold) and MACD crosses above signal line
            if rsi < 30 and macd > signal and position == 0:
                shares = balance // price
                cost = shares * price
                if cost <= balance:
                    balance -= cost
                    position += shares
                    trades.append(('Buy', df.index[i], price, shares))
            
            # Sell signal: RSI > 70 (overbought) or MACD crosses below signal line
            elif (rsi > 70 or macd < signal) and position > 0:
                revenue = position * price
                balance += revenue
                trades.append(('Sell', df.index[i], price, position))
                position = 0
    
    # Calculate final portfolio value
    final_value = balance + position * df['Close'].iloc[-1]
    return trades, final_value

# Main function
async def main():
    # Parameters
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2025-06-14'
    look_back = 60
    
    # Fetch and prepare data
    df = fetch_stock_data(ticker, start_date, end_date)
    df = calculate_indicators(df)
    
    # Prepare LSTM data
    X, y, scaler = prepare_lstm_data(df, look_back)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train model
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Adjust dataframe for predictions
    df_test = df.iloc[train_size + look_back:].copy()
    df_test['Predictions'] = predictions[:len(df_test)]
    
    # Execute trading strategy
    trades, final_value = trading_strategy(df_test, predictions)
    
    # Print results
    print(f"\nStock Trading AI Results for {ticker}")
    print(f"Initial Balance: $10000")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print("\nTrades Executed:")
    for trade in trades:
        print(f"{trade[0]} at {trade[1]}: {trade[3]} shares @ ${trade[2]:.2f}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
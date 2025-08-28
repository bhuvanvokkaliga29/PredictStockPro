import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start="2009-01-01", end="2025-01-01"):
    # Explicitly turn off auto_adjust to get actual close prices
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    return df

def preprocess_train_test(df):
    train_size = int(len(df) * 0.85)
    train_df = df['Close'][:train_size].dropna()
    test_df = df['Close'][train_size:].dropna()
    return train_df, test_df

def scale_data(train_df, test_df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df.values.reshape(-1, 1))
    past_100 = train_df.tail(100)
    final_df = pd.concat([past_100, test_df], ignore_index=True).dropna()
    input_data = scaler.transform(final_df.values.reshape(-1, 1))
    return scaler, train_scaled, input_data

def prepare_test_sequences(input_data):
    import numpy as np
    x_test = []
    y_test = []
    for i in range(100, len(input_data)):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_test, y_test

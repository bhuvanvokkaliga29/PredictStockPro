import matplotlib.pyplot as plt

def plot_closing_prices_ma(df):
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['Close'], label='Closing Price', color='red')
    ax.plot(ma100, label='100-day MA', color='blue')
    ax.plot(ma200, label='200-day MA', color='green')
    ax.set_title('Closing Price with Moving Averages')
    ax.legend()
    return fig

def plot_prediction_vs_actual(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(y_test, label='Actual Price', color='green')
    ax.plot(y_pred, label='Predicted Price', color='red')
    ax.set_title('Predicted vs Actual Closing Prices')
    ax.legend()
    return fig

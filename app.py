import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="PredictStockPro 💹", layout="wide")

# Black & White Theme CSS
st.markdown("""
<style>
.stApp {
  background-color: #000000;
  color: #FFFFFF;
  min-height: 100vh;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
/* Sidebar background and text */
[data-testid="stSidebar"] {
  background-color: #111111;
  color: #FFFFFF;
  padding: 20px;
  border-radius: 12px;
}
/* Floating cards style */
.floating-card {
  background-color: #222222 !important;
  border-radius: 16px;
  padding: 25px;
  margin-bottom: 25px;
  border: 1px solid #555555;
  box-shadow: 0 8px 15px rgba(255,255,255,0.05);
  transition: transform 0.3s ease;
  color: #FFFFFF !important;
}
.floating-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 24px rgba(255,255,255,0.15);
}
/* Buttons style */
.stButton>button {
  background-color: #000000 !important;
  color: #ffffff !important;
  border: 1px solid #ffffff !important;
  font-weight: 700 !important;
  border-radius: 12px !important;
  padding: 14px 40px !important;
  font-size: 1.1em !important;
  transition: background-color 0.3s ease !important;
}
.stButton>button:hover {
  background-color: #333333 !important;
  border-color: #ffffff !important;
}
/* Highlighted header */
.highlight-title {
  font-size: 3em;
  font-weight: 800;
  text-align: center;
  color: #ffffff !important;
  background-color: #111111;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 30px;
}
/* Metric boxes styling */
.css-1kyxreq {
  background-color: #444444 !important;
  border-radius: 14px !important;
  padding: 15px !important;
  font-weight: 600 !important;
  color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown('<div class="highlight-title">PredictStockPro 💹</div>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; margin-bottom:40px;'>AI-powered Stock Price Forecasting Made Effortless</h3>", unsafe_allow_html=True)

# Initialize ticker in session state
if 'ticker' not in st.session_state:
    st.session_state.ticker = "AAPL"

with st.sidebar:
    st.markdown('<div class="floating-card">', unsafe_allow_html=True)
    st.header("🔎 Configure Your Analysis")
    # Remove selectbox; only manual input
    ticker_manual = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", value=st.session_state.ticker).strip().upper()
    if ticker_manual and ticker_manual != st.session_state.ticker:
        st.session_state.ticker = ticker_manual
    start_date = st.date_input("Start Date", datetime.date(2018, 1, 1), min_value=datetime.date(2000, 1, 1), max_value=datetime.date.today())
    end_date = st.date_input("End Date (max today)", datetime.date.today(), min_value=datetime.date(2000, 1, 1), max_value=datetime.date.today())
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="margin-top:20px; margin-bottom:60px;">', unsafe_allow_html=True)
    run = st.button("🚀 Run Prediction")
    st.markdown('</div>', unsafe_allow_html=True)

if not run:
    st.info("Please enter a stock ticker, select date range, then click 'Run Prediction'.")

if run and st.session_state.ticker:
    ticker = st.session_state.ticker
    try:
        with st.spinner(f"Fetching data for {ticker} & running prediction..."):
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty or df['Close'].dropna().empty:
                st.error("No data found for this ticker and date range. Please try different inputs.")
                st.stop()

        st.success(f"Loaded data for {ticker} from {start_date} to {end_date}")

        # Data description card
        st.markdown('<div class="floating-card">', unsafe_allow_html=True)
        st.subheader(f"{ticker} - Data Summary")
        st.write(df.describe())
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="floating-card">', unsafe_allow_html=True)
            st.subheader("📈 Closing Price Over Time")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df['Close'].dropna(), color='#6e8efb', linewidth=2)
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Price ($)")
            ax1.set_title(f"{ticker} Closing Price")
            st.pyplot(fig1)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="floating-card">', unsafe_allow_html=True)
            st.subheader("🔄 Moving Averages (50 & 200 days)")
            ma50 = df['Close'].rolling(window=50).mean().dropna()
            ma200 = df['Close'].rolling(window=200).mean().dropna()
            close = df['Close'].dropna()
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(close, alpha=0.6, label='Close', color='#a777e3', linewidth=2)
            ax2.plot(ma50, label='50-day MA', color='#d7a0f4', linewidth=2)
            ax2.plot(ma200, label='200-day MA', color='#6e8efb', linewidth=2)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price ($)")
            ax2.legend()
            st.pyplot(fig2)
            st.markdown('</div>', unsafe_allow_html=True)

        # Prepare data for prediction
        train_data = df['Close'][:int(len(df) * 0.85)]
        test_data = df['Close'][int(len(df) * 0.85):]
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(np.array(train_data).reshape(-1, 1))
        model = load_model('keras_model.h5')
        last_100_days = train_data.tail(100)
        full_scaled_input = pd.concat([last_100_days, test_data], ignore_index=True)
        scaled_data = scaler.transform(np.array(full_scaled_input).reshape(-1, 1))
        x_test = []
        y_test = []
        for i in range(100, len(scaled_data)):
            x_test.append(scaled_data[i - 100:i, 0])
            y_test.append(scaled_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_pred_scaled = model.predict(x_test)
        scale_factor = 1 / scaler.scale_[0]
        y_pred = y_pred_scaled * scale_factor
        y_true = y_test * scale_factor
        st.markdown('<div class="floating-card">', unsafe_allow_html=True)
        st.subheader("🤖 Predicted vs Actual Closing Prices")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        ax3.plot(y_true, 'g', label='Actual', linewidth=2)
        ax3.plot(y_pred, 'r', label='Predicted', linestyle='dashed', linewidth=2)
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Price ($)")
        ax3.legend()
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        col3, col4 = st.columns([3, 1])
        with col3:
            st.markdown('<div class="floating-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Performance Metrics")
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("MAE", f"{mae:.4f}")
            st.metric("R² Score", f"{r2:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            pred_df = pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()})
            csv_file = pred_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_file,
                file_name=f"{ticker}_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"An error occurred: {e}")

# PredictStockPro
PredictStockPro is an AI-powered stock forecasting app built with Streamlit and LSTM models. It enables users to input tickers, select date ranges, and visualize actual vs predicted prices with metrics and CSV export, making financial forecasting simple and interactive.

PredictStockPro – AI-Powered Stock Price Forecasting 💹

An advanced web application for stock price prediction, built with Streamlit and LSTM-based neural networks.

About the Project

PredictStockPro is designed to forecast future stock closing prices using historical market data and deep learning models.
It offers an intuitive interface where users can input stock tickers manually, set custom date ranges, and visualize both historical and predicted prices in an interactive manner.

Key Features

Manual ticker input for maximum flexibility

Interactive charts with historical closing prices and moving averages

LSTM neural network predictions of stock closing prices

Performance metrics display (RMSE, MAE, R²)

Option to download predictions as CSV

Minimal, high-contrast UI with floating card design

Personal Introduction

I am Bhuvan Gowda H K, a CSE - AIML student from AMC Engineering College, Bengaluru.
My interests span Artificial Intelligence, Machine Learning, Full-Stack Development, and Competitive Programming.

Currently working with the Trust Builders team on projects involving full-stack solutions, AI/ML applications, web design, and media innovations.

Skilled in Python, C, JavaScript, HTML, CSS, React.js, Node.js, MongoDB, and SQL, with a continuous focus on AI/ML advancements.

Learning and mastering DSA in Python and C++, with a long-term vision of pursuing M.Tech in Artificial Intelligence abroad.

Actively contributing to open-source projects and sharing work with the developer community.

Running the Application Locally

Prerequisites

Python 3.7 or higher

pip package manager

Steps

Clone the repository or download the source code.

Place the trained LSTM model file keras_model.h5 inside the application directory.

Install dependencies:

pip install streamlit yfinance numpy pandas matplotlib scikit-learn tensorflow keras


Launch the app:

streamlit run app.py


Open the application in your browser (default: http://localhost:8501).

Usage Guide

Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL).

Choose a custom date range (default: 2018 – present).

Run the prediction.

Explore visualizations of historical and forecasted prices.

Check performance metrics (RMSE, MAE, R²).

Export results as CSV for extended analysis.

Acknowledgements

Yahoo Finance for market data

Streamlit for app framework

TensorFlow & Keras for deep learning models

scikit-learn, pandas, numpy, matplotlib, plotly for data processing and visualization

License

This project is released under the MIT License.
You are free to use, modify, and distribute with attribution.

Maintained by: Bhuvan Gowda H K

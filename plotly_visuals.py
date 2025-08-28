import plotly.graph_objects as go
import pandas as pd

def plotly_closing_price(df, ticker):
    df_plot = df.copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    # Plot original close prices, not scaled
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot['Close'], 
        mode='lines', 
        name='Close Price', 
        line=dict(color='royalblue', width=2)
    ))
    fig.update_layout(
        title=f"{ticker} Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        template='plotly_dark',
        height=500
    )
    return fig

def plotly_prediction_vs_actual(y_test, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=y_test.flatten(), 
        mode='lines', 
        name='Actual Price', 
        line=dict(color='limegreen', width=2)
    ))
    fig.add_trace(go.Scatter(
        y=y_pred.flatten(), 
        mode='lines', 
        name='Predicted Price', 
        line=dict(color='tomato', width=2, dash='dash')
    ))
    fig.update_layout(
        title="Predicted vs Actual Closing Prices",
        xaxis_title="Time Step",
        yaxis_title="Price",
        hovermode='x unified',
        template='plotly_dark',
        height=500
    )
    return fig

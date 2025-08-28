import streamlit as st
import pandas as pd
from io import BytesIO

def convert_df_to_csv(df: pd.DataFrame) -> BytesIO:
    csv = df.to_csv(index=False).encode('utf-8')
    return BytesIO(csv)

def add_download_button(df: pd.DataFrame, label: str = "Download CSV"):
    csv_bytes = convert_df_to_csv(df)
    st.sidebar.download_button(
        label=label,
        data=csv_bytes,
        file_name='predictions.csv',
        mime='text/csv'
    )

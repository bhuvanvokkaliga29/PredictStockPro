import yfinance as yf
from pandas_datareader import data as pdr

yf.pdr_override()
df = pdr.get_data_yahoo('GOOGL', start='2020-01-01', end='2020-12-31')
print(df.head())

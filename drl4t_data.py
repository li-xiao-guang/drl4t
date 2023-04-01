import pandas as pd
import yfinance as yf
import datetime as dt

def download(csv_file):
    df = pd.read_csv(csv_file)

    data = {}
    for symbol in df['Symbol']:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(interval='1d', period='5y', actions=False)
            if len(hist) > 0:
                data[symbol] = hist
        except:
            print(f'Failed to download data for {symbol}')

    for symbol, hist in data.items():
        hist['SMA10'] = hist['Close'].rolling(window=10).mean()
        hist['SMA20'] = hist['Close'].rolling(window=20).mean()

        ema_short = hist['Close'].ewm(span=12, adjust=False).mean()
        ema_long = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['MACD_DIF'] = ema_short - ema_long
        hist['MACD_SIGNAL'] = hist['MACD_DIF'].ewm(span=9, adjust=False).mean()
        hist['MACD'] = hist['MACD_DIF'] - hist['MACD_SIGNAL']

        sma = hist['Close'].rolling(window=20).mean()
        std = hist['Close'].rolling(window=20).std()
        hist['UB'] = sma + 2 * std
        hist['LB'] = sma - 2 * std

        mfm = ((hist['Close'] - hist['Low']) - (hist['High'] - hist['Close'])) / (hist['High'] - hist['Low'])
        mfv = mfm * hist['Volume']
        hist['CMF'] = mfv.rolling(21).sum() / hist['Volume'].rolling(21).sum()

        hist['SMARatio10'] = hist['Close'] / hist['SMA10']
        hist['SMARatio20'] = hist['Close'] / hist['SMA20']

        hist['BBP'] = (hist['Close'] - hist['LB']) / (hist['UB'] - hist['LB'])

        data[symbol] = hist.dropna()

    train_data = {}
    test_data = {}

    split_date = dt.date.today() - pd.DateOffset(days=100)

    for symbol, hist in data.items():
        train_data[symbol] = hist[:split_date - pd.Timedelta(days=1)]
        test_data[symbol] = hist[split_date:]
        
    return train_data, test_data
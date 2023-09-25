import numpy as np


def add_metrics(dataframe):
    dataframe.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    dataframe['Log Return'] = np.log(dataframe['Close'] / dataframe['Close'].shift(1)) * 100

    dataframe['Price Change'] = dataframe['Close'].diff()

    dataframe['Percent Change'] = dataframe['Close'].pct_change() * 100

    dataframe['SMA 5'] = (dataframe['Close'].rolling(window=5).mean() / dataframe['Close']) * 100
    dataframe['SMA 14'] = (dataframe['Close'].rolling(window=14).mean() / dataframe['Close']) * 100
    dataframe['SMA 20'] = (dataframe['Close'].rolling(window=20).mean() / dataframe['Close']) * 100

    dataframe['Gain'] = dataframe['Price Change'].apply(lambda x: x if x > 0 else 0)
    dataframe['Loss'] = dataframe['Price Change'].apply(lambda x: -x if x < 0 else 0)

    dataframe['Avg Gain 5'] = dataframe['Gain'].rolling(window=5).mean()
    dataframe['Avg Loss 5'] = dataframe['Loss'].rolling(window=5).mean()

    dataframe['Avg Gain 14'] = dataframe['Gain'].rolling(window=14).mean()
    dataframe['Avg Loss 14'] = dataframe['Loss'].rolling(window=14).mean()

    dataframe['Avg Gain 20'] = dataframe['Gain'].rolling(window=20).mean()
    dataframe['Avg Loss 20'] = dataframe['Loss'].rolling(window=20).mean()

    dataframe['RS 5'] = dataframe['Avg Gain 5'] / dataframe['Avg Loss 5']
    dataframe['RS 14'] = dataframe['Avg Gain 14'] / dataframe['Avg Loss 14']
    dataframe['RS 20'] = dataframe['Avg Gain 20'] / dataframe['Avg Loss 20']

    dataframe['RSI 5'] = 100 - (100 / (1 + dataframe['RS 5']))
    dataframe['RSI 14'] = 100 - (100 / (1 + dataframe['RS 14']))
    dataframe['RSI 20'] = 100 - (100 / (1 + dataframe['RS 20']))

    dataframe['EMA_12'] = dataframe['Close'].ewm(span=12).mean()
    dataframe['EMA_26'] = dataframe['Close'].ewm(span=26).mean()

    dataframe['MACD'] = dataframe['EMA_12'] - dataframe['EMA_26']

    dataframe['Signal Line'] = dataframe['MACD'].ewm(span=9).mean()
    dataframe['Signal'] = np.where(dataframe['MACD'] > dataframe['Signal Line'], 1, 0)

    dataframe = dataframe.drop(['Open', 'High', 'Low',
                                'Avg Gain 5', 'Avg Loss 5',
                                'Avg Gain 14', 'Avg Loss 14',
                                'Avg Gain 20', 'Avg Loss 20',
                                'RS 5', 'RS 14', 'RS 20',
                                'Gain', 'Loss',
                                'MACD', 'Signal Line', 'EMA_12', 'EMA_26', 'Price Change'
                                ], axis=1)

    dataframe = dataframe.iloc[26:]
    return dataframe

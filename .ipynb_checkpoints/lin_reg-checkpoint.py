from datetime import datetime as dt
import numpy as np
import pandas as pd
import talib


data_df = pd.read_csv('/Users/snehadeepguha/Downloads/reddy_daily.csv', index_col=[0]).drop('co_code', axis=1)
data_df['date'] = data_df['date'].map(lambda x: dt.strptime(x, '%m/%d/%Y %I:%M:%S %p').date())
data_df['volume'] = data_df['volume'].astype(float)
data_df['target'] = np.append(data_df['open'][1:].values, [np.nan])
data_df = data_df.set_index('date')

# Simple Moving Average
data_df['sma_5'] = talib.SMA(data_df['close'].values, timeperiod=5)
data_df['sma_10'] = talib.SMA(data_df['close'].values, timeperiod=10)

# Exponential Moving Average
data_df['ema_20'] = talib.EMA(data_df['close'].values, timeperiod=20)

# Momentum 6 Month / Momentum 12 Month
data_df['mtm6_mtm12'] = talib.MOM(data_df['close'].values, timeperiod=126)/talib.MOM(data_df['close'].values, 
                                  timeperiod=252)

# Stochastic Relative Strength Index
data_df['fastk'], data_df['fastd'] = talib.STOCHRSI(data_df['close'].values, timeperiod=14, fastk_period=5,
                                                    fastd_period=3, fastd_matype=0)

# Rate Of Change
data_df['roc_10'] = talib.ROC(data_df['close'].values, timeperiod=10)

# Bollinger Bands
data_df['bband_upper'], data_df['bband_middle'], data_df['bband_lower'] = talib.BBANDS(data_df['close'].values,
                                                                                     timeperiod=5, nbdevup=2, nbdevdn=2,
                                                                                     matype=0)

# Moving Average Convergence Divergence
data_df['macd'], data_df['macdsignal'], data_df['macdhist'] = talib.MACD(data_df['close'].values, fastperiod=12, 
                                                                         slowperiod=26, signalperiod=9)

# On Balance Volume
data_df['obv'] = talib.OBV(data_df['close'].values, data_df['volume'].values)

# Commodity Channel Index
data_df['cci_14'] = talib.CCI(data_df['high'].values, data_df['low'].values, data_df['close'].values, timeperiod=14)


# Average True Range
data_df['atr_14'] = talib.ATR(data_df['high'].values, data_df['low'].values, data_df['close'].values, timeperiod=14)

data_df = data_df.dropna(axis=0, how='any')


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


target = data_df.pop('target')
X_train, X_test, y_train, y_test = train_test_split(data_df, target, test_size=0.20, shuffle=False)
lin_reg = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
lin_reg.fit(X_train, y_train)
lin_reg.score(X_train, y_train)
lin_reg.score(X_test, y_test)
target = pd.DataFrame(target)
target['pred'] = lin_reg.predict(data_df)
target = target.join(data_df[['open', 'high', 'low', 'close']])
target = target[['open', 'high', 'low', 'close', 'target', 'pred']]
target.loc[:, 'accuracy'] = np.nan
for i in range(len(target)):
    if (target.iloc[i, 4] > target.iloc[i, 3]) and (target.iloc[i, 5] > target.iloc[i, 3]):
        target.iloc[i, 6] = 1
    elif (target.iloc[i, 4] < target.iloc[i, 3]) and (target.iloc[i, 5] < target.iloc[i, 3]):
        target.iloc[i, 6] = 1
    else:
        target.iloc[i, 6] = 0
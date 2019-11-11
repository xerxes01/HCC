import pandas as pd
import talib
import numpy as np

FILE_PATH = "./225537_daily.csv"

def get_data(type = "reg"):
    
    data_df = pd.read_csv(FILE_PATH, index_col=[0])
    
    #data_df['date'] = data_df['date'].map(lambda x: dt.strptime(x, '%m/%d/%Y %I:%M:%S %p').date())
    data_df['volume'] = data_df['volume'].astype(float)
    #data_df = data_df.set_index('date')

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
    # Chaikin A/D Oscillator
    data_df['adosc'] = talib.ADOSC(data_df['high'], data_df['low'], data_df['close'], data_df['volume'], fastperiod=3,
                                 slowperiod=10)
    # Commodity Channel Index
    data_df['cci_14'] = talib.CCI(data_df['high'].values, data_df['low'].values, data_df['close'].values, timeperiod=14)
    # Average True Range
    data_df['atr_14'] = talib.ATR(data_df['high'].values, data_df['low'].values, data_df['close'].values, timeperiod=14)

    if type == "reg" :
        # Target
        data_df['target'] = np.append(data_df['close'][1:].values, [np.nan])
        # Drop Rows With NA Values In Any Column
        data_df = data_df.dropna(axis=0, how='any')
        
        # Popping The Target Column
        target = data_df.pop('target').values
    
    if type == "class":
        #### add another column with the class 0 (for downward trajectory) and 1(for positive trajectory)
        data_df['target'] = np.append(data_df['close'][1:].values, [np.nan])
        data_df = data_df.dropna(axis=0, how='any')
        data_df['class'] = np.where((data_df['close'] < data_df['target']),0,1)

        data_df.drop('target',axis = 1, inplace = True)
        
        target= data_df.pop('class').values

    return data_df, target
   
    
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    return(np.array(data), np.array(labels))

def normalise_data(data_df,):
    TRAIN_SPLIT = round(0.8 * len(data_df))
    # Normalizing Data
    dataset = data_df.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset-data_mean)/data_std
    return dataset

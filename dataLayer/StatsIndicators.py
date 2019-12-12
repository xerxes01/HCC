import numpy as np
import pandas as pd
import talib


class StatsIndicators:

    @staticmethod
    def get_all_indicators(data_df):
        indicators = pd.concat([StatsIndicators.get_sma(data_df),
                                StatsIndicators.get_ema(data_df),
                                StatsIndicators.get_momentum(data_df),
                                StatsIndicators.get_stochastic_rsi(data_df),
                                StatsIndicators.get_rate_of_change(data_df),
                                StatsIndicators.get_bollinger_bands(data_df),
                                StatsIndicators.get_macd(data_df),
                                StatsIndicators.get_ad_oscillator(data_df),
                                StatsIndicators.get_commodity_channel_index(data_df),
                                StatsIndicators.get_fft(data_df),
                                StatsIndicators.get_average_true_range(data_df)], axis=1, sort=False)
        indicators = indicators[:data_df.shape[0]]
        print("Shape of the dataframe with indicators: {}".format(indicators.shape))
        indicators['dev_sma_5'] = np.log(data_df['close']) - np.log(indicators['sma_5'])
        indicators['dev_sma_10'] = np.log(data_df['close']) - np.log(indicators['sma_10'])
        indicators['dev_ema_20'] = np.log(data_df['close']) - np.log(indicators['ema_20'])
        indicators['dev_fft_6'] = np.log(data_df['close']) - np.log(indicators['fft_6'])
        indicators['dev_fft_10'] = np.log(data_df['close']) - np.log(indicators['fft_10'])
        indicators['dev_fft_20'] = np.log(data_df['close']) - np.log(indicators['fft_20'])
        indicators['dev_fft_50'] = np.log(data_df['close']) - np.log(indicators['fft_50'])
        print("Shape of the dataframe after adding deviations: {}".format(indicators.shape))
        return indicators

    @staticmethod
    def get_sma(data_df, time_periods=[5, 10], to_apply='close'):
        output = pd.DataFrame()
        for i in time_periods:
            output['sma_' + str(i)] = talib.SMA(data_df[to_apply].values, timeperiod=i)
        print("----- Done SMA: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_ema(data_df, time_periods=[20], to_apply='close'):
        output = pd.DataFrame()
        for i in time_periods:
            output['ema_' + str(i)] = talib.EMA(data_df[to_apply].values, timeperiod=i)
        print("----- Done EMA: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_momentum(data_df, time_periods=[(6, 12)], to_apply='close'):
        output = pd.DataFrame()
        num_work_days_a_month = 21
        for i in time_periods:
            mom_start = talib.MOM(data_df[to_apply].values, timeperiod=num_work_days_a_month * i[0])
            mom_end = talib.MOM(data_df[to_apply].values, timeperiod=num_work_days_a_month * i[1])
            output['mtm' + str(i[0]) + "_mtm" + str(i[1])] = mom_start / mom_end
        print("----- Done Momentum: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_stochastic_rsi(data_df, time_periods=[14]):
        output = pd.DataFrame()
        for i in time_periods:
            output['fastk'], output['fastd'] = talib.STOCHRSI(data_df['close'].values, timeperiod=i, fastk_period=5,
                                                              fastd_period=3, fastd_matype=0)
        print("----- Done Stochastic RSI: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_rate_of_change(data_df, time_periods=[10]):
        output = pd.DataFrame()
        for i in time_periods:
            output['roc_' + str(i)] = talib.ROC(data_df['close'].values, timeperiod=i)
        print("----- Done ROC: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_bollinger_bands(data_df, time_periods=[5]):
        output = pd.DataFrame()
        for i in time_periods:
            output['bband_upper'], output['bband_middle'], output['bband_lower'] = talib.BBANDS(
                data_df['close'].values, timeperiod=i, nbdevup=2, nbdevdn=2, matype=0)
        print("----- Done BB: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_macd(data_df, time_periods=[12]):
        output = pd.DataFrame()
        for i in time_periods:
            output['macd'], output['macdsignal'], output['macdhist'] \
                = talib.MACD(data_df['close'].values, fastperiod=i, slowperiod=26, signalperiod=9)
        print("----- Done MACD: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_ad_oscillator(data_df, time_periods=[3]):
        output = pd.DataFrame()
        for i in time_periods:
            output['adosc'] = talib.ADOSC(data_df['high'].values, data_df['low'].values, data_df['close'].values,
                                          data_df['volume'].values,
                                          fastperiod=i, slowperiod=10)
        print("----- Done AD Oscillator: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_commodity_channel_index(data_df, time_periods=[14]):
        output = pd.DataFrame()
        for i in time_periods:
            output['cci_' + str(i)] = talib.CCI(data_df['high'].values, data_df['low'].values, data_df['close'].values,
                                                timeperiod=i)
        print("----- Done CCI: Output Shape {} -----".format(output.shape))
        return output

    @staticmethod
    def get_average_true_range(data_df, time_periods=[14]):
        output = pd.DataFrame()
        for i in time_periods:
            output['atr_' + str(i)] = talib.ATR(data_df['high'].values, data_df['low'].values, data_df['close'].values,
                                                timeperiod=i)
        print("----- Done ATR: Output Shape {} -----".format(output.shape))
        return output
    
    @staticmethod
    def get_fft(data_df, time_periods=[6, 10, 20, 50], to_apply='close'):
        output = pd.DataFrame()
        close_fft = np.fft.fft(np.asarray(data_df[to_apply].tolist()))
        fft_df = pd.DataFrame({'fft':close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
        for i in time_periods:
            fft_list = np.asarray(fft_df['fft'].tolist())
            fft_list[int(i/2):int(-i/2)] = 0
            output.loc[:, 'fft_{}'.format(i)] = np.fft.ifft(fft_list).real
        print("----- Done FFT: Output Shape {} -----".format(output.shape))
        return output



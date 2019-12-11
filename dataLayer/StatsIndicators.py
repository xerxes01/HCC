import pandas as pd
import talib


class StatsIndicators:

    @staticmethod
    def get_all_indicators(data_df):
        indicators = pd.concat([StatsIndicators.get_sma(data_df),StatsIndicators.get_ema(data_df),
                                StatsIndicators.get_momentum(data_df),
                                StatsIndicators.get_stochastic_rsi(data_df),
                                StatsIndicators.get_rate_of_change(data_df),
                                StatsIndicators.get_bollinger_bands(data_df), StatsIndicators.get_macd(data_df),
                                StatsIndicators.get_ad_oscillator(data_df),
                                StatsIndicators.get_commodity_channel_index(data_df),
                                StatsIndicators.get_average_true_range(data_df)], axis=1, sort=False)
        indicators = indicators[:data_df.shape[0]]
        return indicators

    @staticmethod
    def get_sma(data_df, time_periods=[5, 10], to_apply='close'):
        print("sma called")
        output = pd.DataFrame()
        for i in time_periods:
            output['sma_' + str(i)] = talib.SMA(data_df[to_apply].values, timeperiod=i)
        print("output size sma", output.shape)
        return output

    @staticmethod
    def get_ema(data_df, time_periods=[20], to_apply='close'):
        output = pd.DataFrame()
        for i in time_periods:
            output['ema_' + str(i)] = talib.EMA(data_df[to_apply].values, timeperiod=i)
        print("output size ema", output.shape)
        return output

    @staticmethod
    def get_momentum(data_df, time_periods=[(6, 12)], to_apply='close'):
        output = pd.DataFrame()
        num_work_days_a_month = 21
        for i in time_periods:
            mom_start = talib.MOM(data_df[to_apply].values, timeperiod=num_work_days_a_month * i[0])
            mom_end = talib.MOM(data_df[to_apply].values, timeperiod=num_work_days_a_month * i[1])
            output['mtm' + str(i[0]) + "_mtm" + str(i[1])] = mom_start / mom_end
        print("output size", output.shape)
        return output

    @staticmethod
    def get_stochastic_rsi(data_df, time_periods=[14]):
        output = pd.DataFrame()
        for i in time_periods:
            output['fastk'], output['fastd'] = talib.STOCHRSI(data_df['close'].values, timeperiod=i, fastk_period=5,
                                                              fastd_period=3, fastd_matype=0)
        print("output size", output.shape)

        return output

    @staticmethod
    def get_rate_of_change(data_df, time_periods=[10]):
        output = pd.DataFrame()
        for i in time_periods:
            output['roc_' + str(i)] = talib.ROC(data_df['close'].values, timeperiod=i)
        print("output size", output.shape)
        return output

    @staticmethod
    def get_bollinger_bands(data_df, time_periods=[5]):
        output = pd.DataFrame()
        for i in time_periods:
            output['bband_upper'], output['bband_middle'], output['bband_lower'] = talib.BBANDS(
                data_df['close'].values, timeperiod=i, nbdevup=2, nbdevdn=2, matype=0)
        print("output size", output.shape)

        return output

    @staticmethod
    def get_macd(data_df, time_periods=[12]):
        output = pd.DataFrame()
        for i in time_periods:
            output['macd'], output['macdsignal'], output['macdhist'] \
                = talib.MACD(data_df['close'].values, fastperiod=i, slowperiod=26, signalperiod=9)
        print("output size", output.shape)

        return output

    @staticmethod
    def get_ad_oscillator(data_df, time_periods=[3]):
        output = pd.DataFrame()
        for i in time_periods:
            output['adosc'] = talib.ADOSC(data_df['high'].values, data_df['low'].values, data_df['close'].values,
                                          data_df['volume'].values,
                                          fastperiod=i, slowperiod=10)
        print("output size", output.shape)

        return output

    @staticmethod
    def get_commodity_channel_index(data_df, time_periods=[14]):
        output = pd.DataFrame()
        for i in time_periods:
            output['cci_' + str(i)] = talib.CCI(data_df['high'].values, data_df['low'].values, data_df['close'].values,
                                                timeperiod=i)
        print("output size", output.shape)

        return output

    @staticmethod
    def get_average_true_range(data_df, time_periods=[14]):
        output = pd.DataFrame()
        for i in time_periods:
            output['atr_' + str(i)] = talib.ATR(data_df['high'].values, data_df['low'].values, data_df['close'].values,
                                                timeperiod=i)
        print("output size", output.shape)

        return output

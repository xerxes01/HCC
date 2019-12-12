from dataLayer.DataSource import DataSource
from dataLayer.StatsIndicators import StatsIndicators
import numpy as np
import pandas as pd
import util.DataUtil as data_util
import util.Constants as Constants


class DataLayer:

    def __init__(self, log_return=True):
        self.is_log_return = log_return
        self.__data_source = DataSource()
        self.__original_df = self.__data_source.read_data()
        self.__original_df.reset_index(inplace=True)
        self.__create_data_with_indicators()
        self.__clean_processed_data()
        print("Shape after adding all indicators: {}".format(self.__data_with_indicators.shape))
        if log_return:
            self.__create_log_ret()
        print("Shape after adding LR: {}".format(self.__data_with_indicators.shape))
        self.__data_with_indicators = data_util.normalise_data(self.__data_with_indicators)
        self.__regression_target = self.__create_regression_data()
        print("Shape of Regression Target: {}".format(self.__regression_target.shape))
        self.__classification_target = self.__create_classification_data()
        print("Shape of Classification Target: {}".format(self.__classification_target.shape))

    def __create_data_with_indicators(self):
        indicators = StatsIndicators.get_all_indicators(self.__original_df)
        self.__data_with_indicators = pd.concat([self.__original_df, indicators], axis=1, sort=False)
        self.__data_with_indicators.set_index('date', inplace=True)
        self.__original_df.set_index('date', inplace=True)

    def __clean_processed_data(self):
        self.__data_with_indicators = self.__data_with_indicators.dropna(axis=0, how='any')
    
    def __create_log_ret(self):
        self.__data_with_indicators['open_lr'] = np.log(self.__data_with_indicators['open']) - np.log(self.__data_with_indicators['open'].shift(1))
        self.__data_with_indicators['high_lr'] = np.log(self.__data_with_indicators['high']) - np.log(self.__data_with_indicators['high'].shift(1))
        self.__data_with_indicators['low_lr'] = np.log(self.__data_with_indicators['low']) - np.log(self.__data_with_indicators['low'].shift(1))
        self.__data_with_indicators['close_lr'] = np.log(self.__data_with_indicators['close']) - np.log(self.__data_with_indicators['close'].shift(1))
        self.__data_with_indicators['volume_lr'] = np.log(self.__data_with_indicators['volume']) - np.log(self.__data_with_indicators['volume'].shift(1))
        self.__data_with_indicators['sma_5_lr'] = np.log(self.__data_with_indicators['sma_5']) - np.log(self.__data_with_indicators['sma_5'].shift(1))
        self.__data_with_indicators['sma_10_lr'] = np.log(self.__data_with_indicators['sma_10']) - np.log(self.__data_with_indicators['sma_10'].shift(1))
        self.__data_with_indicators['ema_20_lr'] = np.log(self.__data_with_indicators['ema_20']) - np.log(self.__data_with_indicators['ema_20'].shift(1))
        self.__data_with_indicators['bband_upper_lr'] = np.log(self.__data_with_indicators['bband_upper']) - np.log(self.__data_with_indicators['bband_upper'].shift(1))
        self.__data_with_indicators['bband_middle_lr'] = np.log(self.__data_with_indicators['bband_middle']) - np.log(self.__data_with_indicators['bband_middle'].shift(1))
        self.__data_with_indicators['bband_lower_lr'] = np.log(self.__data_with_indicators['bband_lower']) - np.log(self.__data_with_indicators['bband_lower'].shift(1))
        self.__data_with_indicators['fft_6_lr'] = np.log(self.__data_with_indicators['fft_6']) - np.log(self.__data_with_indicators['fft_6'].shift(1))
        self.__data_with_indicators['fft_10_lr'] = np.log(self.__data_with_indicators['fft_10']) - np.log(self.__data_with_indicators['fft_10'].shift(1))
        self.__data_with_indicators['fft_20_lr'] = np.log(self.__data_with_indicators['fft_20']) - np.log(self.__data_with_indicators['fft_20'].shift(1))
        self.__data_with_indicators['fft_50_lr'] = np.log(self.__data_with_indicators['fft_50']) - np.log(self.__data_with_indicators['fft_50'].shift(1))
        self.__data_with_indicators = self.__data_with_indicators.drop(['open', 'high', 'low', 'close', 'volume', 'sma_5', 'sma_10', 'ema_20', 'fft_6', 'fft_10',
                                                                        'fft_20', 'fft_50'], axis=1).dropna(axis=0, how='any')

    def __create_regression_data(self, shift=1):
        """
        Return train_data with regression target
        :param shift: Target(Prediction) on the basis of <shift> num of days. For eg. If shift is 1, next day value is predicted
        :return: regression target, training data with indicators
        """
        if self.is_log_return:
            self.__data_with_indicators['target'] = np.append(self.__data_with_indicators['close_lr'].values[shift:], [np.nan])
        else:
            self.__data_with_indicators['target'] = np.append(self.__data_with_indicators['close'].values[shift:], [np.nan])

        # Drop Rows With NA Values In Any Column
        self.__data_with_indicators = self.__data_with_indicators.dropna(axis=0, how='any')
        target = self.__data_with_indicators.pop('target').values
        return target

    def __create_classification_data(self, shift=1):
        """
        Return train_data with classification target
        :param shift: Target(Prediction) on the basis of <shift> num of days. For eg. If shift is 1, next day value is predicted
        :return: classification target, training data with indicators
        """
        self.__data_with_indicators['target'] = np.append(self.__data_with_indicators['close_lr'].values[shift:], [np.nan])
        # Drop Rows With NA Values In Any Column
        self.__data_with_indicators['class'] = np.where(self.__data_with_indicators['close_lr'] > 0, 1, 0)
        self.__data_with_indicators.drop('target', axis=1, inplace=True)
        self.__data_with_indicators = self.__data_with_indicators.dropna(axis=0, how='any')
        target = self.__data_with_indicators.pop('class').values
        return target

    def get_regression_data(self, shift=1):
        """
        Returns data with indicators along with regression target
        :param shift: If shift is 1, then we are putting prediction(target) fo the next day.
        If it is 2, then for the 2nd day
        :return: x_train, y_train, x_test, y_test
        """
        if shift == 1:
            return data_util.split_train_test(self.__data_with_indicators, self.__regression_target,
                                              Constants.SPLIT_TRAIN_RATIO)
        else:
            return data_util.split_train_test(self.__data_with_indicators, self.__create_regression_data(shift),
                                              Constants.split_train_ratio)

    def get_classification_data(self, shift=1):
        """
        Returns data with indicators along with classification target
        :param shift: If shift is 1, then we are putting prediction(target) fo the next day.
        If it is 2, then for the 2nd day
        :return: x_train, y_train, x_test, y_test
        """
        if shift == 1:
            return data_util.split_train_test(self.__data_with_indicators, self.__regression_target,
                                              Constants.SPLIT_TRAIN_RATIO)
        else:
            return data_util.split_train_test(self.__data_with_indicators, self.__create_regression_data(shift),
                                              Constants.split_train_ratio)

    def get_regression_target(self):
        return self.__regression_target

    def get_classification_target(self):
        return self.__classification_target

    def get_raw_data(self):
        return self.__original_df

    def get_data_with_indicators(self):
        return self.__data_with_indicators

import pandas as pd
import talib
import numpy as np

FILE_PATH = "./225537_daily.csv"

class Util:

    def __init__(self):
        print("util file created")




    def train_test_data_for_cnn(self, dataset, target, start_index, end_index, history_size, target_size, step,
                                single_step=False):
        TRAIN_SPLIT = round(0.8 * len(dataset))
        past_history = 21
        future_target = 1
        STEP = 1
        X_train, y_train = util.multivariate_data(dataset, target_class, 0, TRAIN_SPLIT, past_history, future_target,
                                                  STEP, single_step=True)
        X_test, y_test = util.multivariate_data(dataset, target_class, TRAIN_SPLIT, None, past_history, future_target,
                                                STEP, single_step=True)
        X_train = X_train.reshape(X_train.shape[0], 21, 21, 1)
        X_test = X_test.reshape(X_test.shape[0], 21, 21, 1)
        return X_train, y_train, X_test, y_test

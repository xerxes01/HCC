import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ConvLSTM2D, Dense, Dropout, GRU, Input, LSTM
from tensorflow.keras.losses import mae
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


import util.DataUtil as data_util
from dataLayer.DataLayer import DataLayer
import util.Constants as Constants


class ModelLSTM(object):

    def __init__(self):
        self.data_layer = DataLayer()
        self.data = self.data_layer.get_data_with_indicators()
        self.class_target = self.data_layer.get_classification_target()
        self.regression_target = self.data_layer.get_regression_target()

    def prepare_data(self):
        train_data_len = round(Constants.SPLIT_TRAIN_RATIO * len(self.data))
        print("len ", train_data_len, " type = =", type(self.data))
        past_history = 5
        future_target = 1
        step = 1
        x_train, y_train = data_util.multivariate_data(self.data.values, self.regression_target, 0, train_data_len,
                                                       past_history, future_target, step, single_step=True)
        x_test, y_test = data_util.multivariate_data(self.data.values, self.regression_target, train_data_len, None,
                                                     past_history, future_target, step, single_step=True)
        return(x_train, y_train, x_test, y_test)

    def create_model(self, shape):
        inputs = Input(shape=shape)
        out_0 = LSTM(2048, activation='tanh', return_sequences=True)(inputs)
        out_1 = LSTM(2048, activation='tanh', return_sequences=True)(out_0)
        out_2 = Dense(2048, activation='tanh')(out_1)
        out_3 = LSTM(2048, activation='tanh')(out_2)
        preds = Dense(1, activation='tanh')(out_3)
        model = Model(inputs=inputs, outputs=preds)
        return(model)

    def compile_model(self, model):
        model.compile(loss="mae", optimizer=Adam())

    def train_model(self, x_train, y_train, x_test, y_test, model):
        print('train model', x_train.shape, y_train.shape)
        batch_size = 32
        epochs = 20

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                  validation_data=(x_test, y_test))
    
    def accuracy_metrics(self, model, y_test, x_test):
        pred_df = pd.DataFrame(columns=['actual', 'predicted'])
        pred_df['actual'] = y_test
        pred_df['predicted'] = model.predict(x_test)
        pred_df['true_positive'] = np.zeros(len(pred_df))
        pred_df['false_positive'] = np.zeros(len(pred_df))
        pred_df['true_negative'] = np.zeros(len(pred_df))
        pred_df['false_negative'] = np.zeros(len(pred_df))
        for i in range(len(pred_df)):
            if (pred_df.iloc[i, 0] < 0) and (pred_df.iloc[i, 1] < 0):
                pred_df.iloc[i, 4] = 1
            elif (pred_df.iloc[i, 0] > 0) and (pred_df.iloc[i, 1] > 0):
                pred_df.iloc[i, 2] = 1
            elif (pred_df.iloc[i, 0] < 0) and (pred_df.iloc[i, 1] > 0):
                pred_df.iloc[i, 3] = 1
            elif (pred_df.iloc[i, 0] > 0) and (pred_df.iloc[i, 1] < 0):
                pred_df.iloc[i, 5] = 1
        print('The number of true positives is {}'.format(sum(pred_df.loc[:, 'true_positive'])))
        print('The number of false positives is {}'.format(sum(pred_df.loc[:, 'false_positive'])))
        print('The number of true negatives is {}'.format(sum(pred_df.loc[:, 'true_negative'])))
        print('The number of false negatives is {}'.format(sum(pred_df.loc[:, 'false_negative'])))

    def run_model(self):
        x_train, y_train, x_test, y_test = self.prepare_data()
        created_model = self.create_model(x_train.shape[-2:])
        self.compile_model(created_model)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        self.train_model(x_train, y_train, x_test, y_test, created_model)
        self.accuracy_metrics(created_model, y_test, x_test)

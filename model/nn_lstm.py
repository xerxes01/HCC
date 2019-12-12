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
        out_0 = Dense(1024, activation='tanh')(inputs)
        out_1 = LSTM(1024, activation='tanh', return_sequences=True)(out_0)
        out_2 = LSTM(1024, activation='tanh')(out_1)
        preds = Dense(1, activation='tanh')(out_2)
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

    def run_model(self):
        x_train, y_train, x_test, y_test = self.prepare_data()
        created_model = self.create_model(x_train.shape[-2:])
        self.compile_model(created_model)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        self.train_model(x_train, y_train, x_test, y_test, created_model)

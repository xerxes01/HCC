from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Flatten, Dropout
import util.DataUtil as data_util
from dataLayer.DataLayer import DataLayer
import util.Constants as Constants


class CNN():

    def __init__(self):
        self.model = self.create_model()
        self.data_layer = DataLayer()
        self.data = self.data_layer.get_data_with_indicators()
        self.class_target = self.data_layer.get_classification_targte()
        self.regression_target = self.data_layer.get_classification_targte()
        self.prepared_data = ""

    def prepare_data(self):
        train_data_len = round(Constants.SPLIT_TRAIN_RATIO * len(self.data))
        print("len ", train_data_len, " type = =", type(self.data))
        past_history = 21
        future_target = 1
        step = 1
        x_train, y_train = data_util.multivariate_data(self.data.values, self.class_target, 0, train_data_len,
                                                       past_history, future_target, step, single_step=True)
        x_test, y_test = data_util.multivariate_data(self.data.values, self.class_target, train_data_len, None,
                                                     past_history,
                                                     future_target, step, single_step=True)
        x_train = x_train.reshape(x_train.shape[0], 21, 21, 1)
        x_test = x_test.reshape(x_test.shape[0], 21, 21, 1)
        return x_train, y_train, x_test, y_test

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(21, 21, 1)))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dropout(0.25))
        model.add(Dense(units=128, activation="relu"))
        # model.add(Dropout(0.5))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))
        return model

    def compile_model(self):
        self.model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    def train_model(self, x_train, y_train, x_test, y_test):
        print('train model', x_train.shape, y_train.shape)
        batch_size = 32
        epochs = 10

        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                       validation_data=(x_test, y_test))

    def run_model(self):
        x_train, y_train, x_test, y_test = self.prepare_data()
        self.create_model()
        self.compile_model()
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        self.train_model(x_train, y_train, x_test, y_test)

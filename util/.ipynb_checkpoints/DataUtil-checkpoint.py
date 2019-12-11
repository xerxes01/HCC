import numpy as np
import pandas as pd
import util.Constants as Constants


def normalise_data(data_frame):
    """
    Normalisation using normal distribution
    This functions normalise the complete dataset (both test and train but parameters are calculated using train data
    and then are used to normalise the test data)
    :param train_split_pct: Percentage of data to be considered for training. Complete data set will be normalised,
     but sigma and mu will be calculated only on basis of training data
    :return: nd array of normalised data
    """
    print("df shape", data_frame.shape)

    train_split = round(Constants.SPLIT_TRAIN_RATIO * len(data_frame))
    # Normalizing Data
    dataset = data_frame.values
    print("data", dataset[:5,])
    data_mean = dataset[:train_split].mean(axis=0)
    data_std = dataset[:train_split].std(axis=0)
    dataset = (dataset - data_mean) / data_std
    return pd.DataFrame(dataset, index=data_frame.index, columns=data_frame.columns)


def split_train_test(data, target, split_train_ratio):
    threshold_point = round(len(data) * split_train_ratio)
    x_train = data[:threshold_point]
    y_train = data[:threshold_point]
    x_test = target[threshold_point:]
    y_test = target[threshold_point:]
    return x_train, y_train, x_test, y_test


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    """
    This function is used to create rolling window.
    :param self:
    :param dataset:
    :param target:
    :param start_index:
    :param end_index:
    :param history_size:
    :param target_size:
    :param step:
    :param single_step:
    :return:
    """

    print(dataset.shape, target.shape, start_index, end_index, history_size, target_size, step, single_step)
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    return (np.array(data), np.array(labels))
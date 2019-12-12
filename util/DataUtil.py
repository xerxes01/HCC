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

    train_split = round(Constants.SPLIT_TRAIN_RATIO * len(data_frame))
    # Normalizing Data
    dataset = data_frame.values
    data_mean = dataset[:train_split].mean(axis=0)
    data_std = dataset[:train_split].std(axis=0)
    dataset = (dataset - data_mean) / data_std
    print('Shape of the normalised datset: {}'.format(dataset.shape))
    return pd.DataFrame(dataset, index=data_frame.index, columns=data_frame.columns)


def split_train_test(data, target, split_train_ratio):
    threshold_point = round(len(data) * split_train_ratio)
    x_train = data[:threshold_point]
    print("Shape of x_train: {}".format(x_train.shape))
    y_train = data[:threshold_point]
    print("Shape of x_train: {}".format(x_train.shape))
    x_test = target[threshold_point:]
    print("Shape of x_train: {}".format(x_train.shape))
    y_test = target[threshold_point:]
    print("Shape of x_train: {}".format(x_train.shape))
    return x_train, y_train, x_test, y_test


def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    """
    This function is used to create rolling window.
    :param self:
    :param dataset: data for which images need to be created
    :param target: images
    :param start_index: 0 if starting from index 0 of dataset
    :param end_index: used for train -test split
    :param history_size: size of the window
    :param target_size: fow how much we want to predict
    :param step: steps size while rolling
    :param single_step: if wants to predict only one day's prediction
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
    print("Shape of multivariate data: {}".format(np.array(data).shape))
    print("Shape of multivariate target: {}".format(np.array(data).shape))
    return (np.array(data), np.array(labels))
import abc


class OurModel(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def prepare_data(self):
        raise NotImplementedError('users must define prepare_data method to use this base class')

    @abc.abstractmethod
    def create_model(self):
        raise NotImplementedError('users must define create_model to use this base class')

    @abc.abstractmethod
    def compile_model(self):
        raise NotImplementedError('users must define compile model to use this base class')

    @abc.abstractmethod
    def train_model(self):
        raise NotImplementedError('users must define train model to use this base class')

    @abc.abstractmethod
    def run_model(self):
        raise NotImplementedError('users must define run model to use this base class')


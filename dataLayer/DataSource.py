import pandas as pd

FILE_PATH = "./test.csv"


class DataSource:

    def read_data(self):
        data_df = pd.read_csv(FILE_PATH, index_col=[0])
        return data_df


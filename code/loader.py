from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
import sklearn.datasets
import pandas as pd

class Loader:
    def __init__(self):
        # Loading datasets 1 & 2, dataset 2 is cardiotocography dataset from openML
        self.dataset_1_x, self.dataset_1_y = sklearn.datasets.load_digits(return_X_y=True)
        self.dataset_2 = fetch_openml(name='cardiotocography', version=2)
        #self.dataset_2 = fetch_openml(name='iris', version=1)
        self.dataset_2_x = self.dataset_2.data
        self.dataset_2_y = self.dataset_2.target.astype('float')
        # self.dataset_2_y = self.dataset_2.target
        self.dataset_2_names = self.dataset_2.feature_names
        print('Dataset Load Complete')

    def numpy_dataset01(self):
        return self.dataset_1_x, self.dataset_1_y

    def numpy_dataset02(self):
        return self.dataset_2_x, self.dataset_2_y

    def numpy_to_pandas(self, np_dataset):
        return pd.DataFrame(data=np_dataset)

    def count_NA_values(self, df):  # df is pandas dataframe
        return df.isnull().sum().sum()

    def min_max_norm(self, dataset):
        if dataset.ndim == 1:
            normalized = (dataset - dataset.min()) / (dataset.max() - dataset.min())
        if dataset.ndim == 2:
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(dataset)
        return normalized


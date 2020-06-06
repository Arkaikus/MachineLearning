import pandas as pd
import numpy as np
import json
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from signal import signal, SIGINT
from sklearn import svm
from utils import read_json,write_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Warehouse:
    def __init__(self,
                 dataframe,
                 inputs=None,
                 target=None,
                 test_size=0.15,
                 random_state=0,
                 shuffle_dataset=False):

        if not isinstance(dataframe, pd.DataFrame):
            raise Exception('Not a Pandas Dataframe')
        if inputs is None:
            raise Exception('Specify the inputs')
        if target is None:
            raise Exception('Specify the target')

        # saving dataframe
        self.df = dataframe

        # input columns
        self.inputs = inputs
        # input len
        self.len_inputs = len(inputs)
        # target column
        self.target = target
        # test set size
        self.test_size = test_size
        # train set random state
        self.random_state = random_state
        # inputs dataframe
        self.X = self.df[inputs]
        # inputs values
        self.X_values = self.X.values
        # output dataframe
        self.y = self.df[target]
        # output values
        self.y_values = self.y.values

        #############################################################
        # Numpy Dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_values,
                                                                                self.y_values,
                                                                                test_size=test_size,
                                                                                random_state=random_state,
                                                                                shuffle=shuffle_dataset)

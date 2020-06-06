# Machine Learning models
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from utils import *

tf.config.list_physical_devices('GPU')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_predictions_flat(train, test):
    train_prediction_length = len(train)
    train_decode = list(train.reshape((train_prediction_length,)))
    test_prediction_length = len(test)
    test_decode = list(test.reshape((test_prediction_length,)))

    y_values = []
    y_values.extend(train_decode)
    y_values.extend(test_decode)
    return y_values


class MLModel:
    def __init__(self, **kwargs):
        self.warehouse = kwargs.get('warehouse',None)
        self.name = kwargs.get('name', '')
        save_dir = kwargs.get('save_dir', None)
        self.save_dir = os.path.join(BASE_DIR, save_dir) if save_dir is not None else None
        self.model_description = {}
        self.model = None
        self.trained = False
        self.train_prediction = []
        self.test_prediction = []
        self.predictions_flat = []
        self.train_rmse = np.inf
        self.test_rmse = np.inf

    def get_model(self, **kwargs):
        pass

    def do_predictions(self):
        if self.model is not None:
            self.train_prediction = self.model.predict(self.warehouse.X_train)
            self.test_prediction = self.model.predict(self.warehouse.X_test)
            self.predictions_flat = get_predictions_flat(self.train_prediction, self.test_prediction)
            self.train_rmse = rmse(self.warehouse.y_train, self.train_prediction)
            self.test_rmse = rmse(self.warehouse.y_test, self.test_prediction)
            self.trained = True

    def get_predictions(self):
        return self.predictions_flat

    def get_rmse(self):
        if self.trained:
            return self.train_rmse, self.test_rmse
        else:
            raise Exception('Untrained Model')

    def save(self, name):
        if self.save_dir is not None and self.trained:
            print("Saving Keras Model")
            location = os.path.join(self.save_dir, name)
            print("Saving to:", location)
            self.model.save(location)
            write_json(os.path.join(location, 'description.json'), self.model_description)

    def load(self, name):
        if self.save_dir is not None:
            print("Loading Keras Model")
            location = os.path.join(self.save_dir, name)
            print('Load from:', location)
            self.model = tf.keras.models.load_model(location)
            self.model_description = read_json(os.path.join(location, 'description.json'))
            self.do_predictions()
            self.trained = True
            return self.model


class DNN(MLModel):
    def get_model(self, **kwargs):
        if self.warehouse is None:
            raise Exception('Warehouse is needed')

        input_dim = kwargs.get('input_dim', None)
        if input_dim is None:
            raise Exception("input_dim can't be None")

        hidden_layers = kwargs.get('hidden_layers', tuple())
        epochs = kwargs.get('epochs', 20)
        shuffle = kwargs.get('shuffle', False)

        self.model_description = {
            'inputs': 'keras.layers.Dense',
            'hidden_layers': hidden_layers,
            'epochs': epochs,
            'activation': 'relu',
            'solver': 'keras.optimizers.Adam'
        }

        self.model = tf.keras.Sequential()
        self.model.add(layers.Dense(input_dim, input_dim=input_dim))

        for units in hidden_layers:
            self.model.add(layers.Dense(units, activation='relu'))

        self.model.add(layers.Dense(1))

        self.model.compile(loss=losses.mean_squared_error, optimizer="adam", metrics=['mean_squared_error'])
        self.model.fit(x=self.warehouse.X_train, y=self.warehouse.y_train, epochs=epochs, verbose=0, shuffle=shuffle)
        super().do_predictions()
        self.trained = True
        return self.model


class CNN(MLModel):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.X_train = kwargs.get('X_train',[])
        self.y_train = kwargs.get('y_train',[])
        self.X_test = kwargs.get('X_test',[])
        self.y_test = kwargs.get('y_test',[])

    def get_model(self, **kwargs):
        input_shape = kwargs.get('input_shape', None)
        if input_shape is None or not isinstance(input_shape, tuple):
            raise Exception("input_shape can't be None, must be tuple")
        epochs = kwargs.get('epochs', 20)
        shuffle = kwargs.get('shuffle', False)
        self.model_description = {
            'inputs': 'keras.layers.Conv2D',
            'activation': 'relu',
            'solver': 'keras.optimizers.Adam'
        }

        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(1, (1, 1), input_shape=input_shape, activation='relu'))
        #self.model.add(layers.MaxPooling2D((2, 2)))
        #self.model.compile(optimizer="adam")
        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        self.history=self.model.fit(x=self.X_train, y=self.y_train, epochs=epochs, verbose=0, shuffle=shuffle)
        self.do_predictions()
        self.trained = True
        return self.model,self.history

    def get_rmse(self):
        pass

    def do_predictions(self):
        self.train_prediction = self.model.predict(self.X_train)
        self.test_prediction = self.model.predict(self.X_test)
        #self.predictions_flat = get_predictions_flat(self.train_prediction, self.test_prediction)
        self.trained = True

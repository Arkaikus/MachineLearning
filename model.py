import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(X,y, shuffle=False, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds

def train_test_ds(X,y, test_size=0.3, batch_size=5, random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state, shuffle=False)
    train_ds = df_to_dataset(X_train,y_train, batch_size=batch_size)
    test_ds = df_to_dataset(X_test,y_test, shuffle=False, batch_size=batch_size)
    
    return (train_ds,test_ds)

def rmse(y_train,y_pred):
    return np.sqrt(mean_squared_error(y_train,y_pred))

def save_model(model,name, params={}):
    # serialize model to JSON
    model_json = model.to_json()
    with open("Modelos/{}.json".format(name), "w") as json_file:
        json_file.write(model_json)
    
    if params:
        with open("Modelos/{}_params.json".format(name), "w") as params_file:
            params_file.write(json.dumps(params))

    # serialize weights to HDF5
    model.save_weights("Modelos/{}.h5".format(name))
    print("Saved model to disk")

def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
class ModeloSecuestros:
    def __init__(self, 
                 csv="", 
                 inputs=['Input_{}'.format(i) for i in range(5)],
                 target='Output_0',
                 test_size = 0.1,
                 random_state = 1
                ):
        
        #cargar pandas dataframe
        self.df = pd.read_csv(csv)
        #columnas de entrada
        self.inputs = inputs
        #columna objetivo
        self.target = target
        #tamaño del conjunto de test
        self.test_size =test_size
        #estado aleatorio del conjunto de entrenamiento
        self.random_state = random_state
        
        #valores del dataframe para columnas de entrada
        self.X = self.df[inputs]
        self.X_values = self.X.values
        #valores del dataframe para columna objetivo
        self.y = self.df[target]
        self.y_values = self.y.values
        
        #############################################################
        # Feature Model
        self.train_ds, self.test_ds = train_test_ds(self.X,self.y, test_size=test_size, random_state=random_state)
        
        #############################################################
        # Sequential Model
        # Tamaño del numero de datos por iteración que se necesitan para entrenar y probar el algoritmo
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_values, self.y_values, test_size=self.test_size, random_state = random_state, shuffle=False)
        
        self.X_train_lstm = self.X_train.reshape((len(self.X_train), len(inputs), 1))
        self.X_test_lstm = self.X_test.reshape((len(self.X_test), len(inputs), 1))
        
        #############################################################
        self.feature_trained = False
        self.sequential_trained = False
        self.lstm_trained= False
        self.clf_trained =False
    def dataframe(self):
        return self.df
    
    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y
    
    def get_test_ds(self):
        return self.test_ds    
    
    def get_X_test(self):
        return self.X_test
    
    def get_y_test(self):
        return self.y_test 
    
    def get_feature_model(self, epochs = 20, shuffle=False, metrics = ['mean_squared_error']):
        feature_columns = []
        for header in self.inputs:
            feature_columns.append(feature_column.numeric_column(header))
        
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self.feature_model = tf.keras.Sequential([
            feature_layer,
            layers.Dense(1, activation='relu')
        ])
        
        self.feature_model.compile(
            optimizer= 'adam',
            loss=losses.mean_squared_error,
            metrics=metrics
        )
        
        self.feature_model.fit(self.train_ds,epochs=epochs, verbose=0, shuffle=shuffle)
        self.feature_train_prediction = self.feature_model.predict(self.train_ds)
        self.feature_prediction = self.feature_model.predict(self.test_ds)
        self.feature_trained = True
        return self.feature_model
    
    def get_rmse_feature(self):
        if self.feature_trained:
            return rmse(self.y_train,self.feature_train_prediction),rmse(self.y_test,self.feature_prediction)
        else:
            return 'Modelo sin inicializar'
    
    def get_scatter(self,train,test):
            train_prediction_length=len(train)
            train_decode = list(train.reshape((train_prediction_length,)))
            test_prediction_length =len(test)
            test_decode = list(test.reshape((test_prediction_length,)))
            
            y_values=[]
            y_values.extend(train_decode)
            y_values.extend(test_decode)           
            
            print("RMSE.",rmse(self.y_values,y_values))
            
            plt.plot(self.y_values, y_values, 'ro')
            plt.plot(self.y_values, self.y_values, 'b-')
            ### Eje x : y_test ##### Eje y: feature_prediction
            min_test,max_test = min(self.y_test), max(self.y_test)
            min_pred,max_pred = min(y_values),max(y_values)
            plt.axis([min_test,max_test,min_pred,max_pred ])
            plt.show()
    
    def get_feature_scatter(self):
        if self.feature_trained:
            self.get_scatter(self.feature_train_prediction,self.feature_prediction);
        else:
            return 'Modelo sin inicializar'
    
    def get_sequential_model(self, input_dim=5,epochs = 20,shuffle=False, metrics=['mean_squared_error']):
        self.sequential_model = tf.keras.Sequential()
        self.sequential_model.add(layers.Dense(np.random.randint(200,400), input_dim=input_dim, activation= "relu"))
        self.sequential_model.add(layers.Dense(np.random.randint(100,200), activation= "relu"))
        self.sequential_model.add(layers.Dense(np.random.randint(5,10), activation= "relu"))
        self.sequential_model.add(layers.Dense(1))
        self.sequential_model.compile(loss=losses.mean_squared_error , optimizer="adam", metrics=metrics)
        self.sequential_model.fit(x=self.X_train, y=self.y_train, epochs=epochs, verbose=0, shuffle=shuffle)
        self.sequential_train_prediction = self.sequential_model.predict(self.X_train)
        self.sequential_prediction = self.sequential_model.predict(self.X_test)
        self.sequential_trained = True
        
        return self.sequential_model
   
    def get_rmse_sequential(self):
        if self.sequential_trained:
            #train,test
            #print(self.sequential_train_prediction)
            return rmse(self.y_train,self.sequential_train_prediction),rmse(self.y_test,self.sequential_prediction)
        else:
            return 'Modelo sin inicializar'
    
    def get_sequential_scatter(self):
        if self.sequential_trained:
            self.get_scatter(self.sequential_train_prediction,self.sequential_prediction);
        else:
            return 'Modelo sin inicializar'
    def plot_feature(self):
        if self.feature_trained:
            plt.plot(self.y_test, self.feature_prediction, 'ro')
            ### Eje x : y_test ##### Eje y: feature_prediction
            plt.axis([min(self.y_test), max(self.y_test), min(self.feature_prediction),max(self.feature_prediction)])
            plt.show()
        else:
            return 'Modelo sin inicializar'
        
    def plot_sequential(self):
        if self.sequential_trained:
            plt.plot(self.y_test, self.sequential_prediction, 'ro')
            ### Eje x : y_test ##### Eje y: feature_prediction
            plt.axis([min(self.y_test), max(self.y_test), min(self.feature_prediction),max(self.feature_prediction)])
            plt.show()
        else:
            return 'Modelo sin inicializar'
    
    def plot_train_feature(self):
        #self.df.plot()
        #plt.plot(self.X_train, self.feature_train_prediction,'r-')
        #plt.show()
        pass
        
    def get_lstm_model(self):
        self.lstm_model = tf.keras.Sequential()
        self.lstm_model.add(layers.LSTM(20, input_shape=(5,1), return_sequences=True))
        self.lstm_model.add(layers.LSTM(20))
        self.lstm_model.add(layers.Dense(1, activation='relu'))
        self.lstm_model.compile(loss=losses.mean_squared_error , optimizer="adam")
        self.lstm_model.fit(x=self.X_train_lstm, y=self.y_train, epochs=20, verbose=0, shuffle=False)
        self.lstm_train_prediction = self.lstm_model.predict(self.X_train_lstm)
        self.lstm_prediction = self.lstm_model.predict(self.X_test_lstm)
        self.lstm_trained = True
        
        return self.lstm_model
    
    def get_rmse_lstm(self):
        if self.lstm_trained:
            #train,test
            return rmse(self.y_train,self.lstm_train_prediction), rmse(self.y_test,self.lstm_prediction)
        else:
            return 'Modelo sin inicializar'
    
    def get_lstm_scatter(self):
        if self.lstm_trained:
            self.get_scatter(self.lstm_train_prediction,self.lstm_prediction);
        else:
            return 'Modelo sin inicializar' 
        
    def get_svm_model(self):        
        self.clf_model = svm.SVC()
        self.clf_model.fit(self.X_train,self.y_train)
        self.clf_train_prediction = self.clf_model.predict(self.X_train)
        self.clf_prediction = self.clf_model.predict(self.X_test)
        self.clf_trained =True
        
        return self.clf_trained
        
    def get_svm_scatter(self):
        if self.clf_trained:
            self.get_scatter(self.clf_train_prediction,self.clf_prediction);
        else:
            return 'Modelo sin inicializar' 
        
        
    def get_rmse_svm(self):
        if self.sequential_trained:
            #train,test
            #print(self.sequential_train_prediction)
            return rmse(self.y_train,self.clf_train_prediction),rmse(self.y_test,self.clf_prediction)
        else:
            return 'Modelo sin inicializar'

        
def run_feature(modelo):
    saved = 0
    notsaved = 0
    for i in range(100):
        print("###########################################################")
        print("Modelo:",i)
        epochs = np.random.randint(20,60)
        print("Epochs:",epochs)
        feature = modelo.get_feature_model(epochs=epochs)
        rmse_train, rmse_test = modelo.get_rmse_feature()
        print('RMSE train:',rmse_train,'RMSE test',rmse_test)
        if rmse_train < 550 and rmse_test <= 850:
            save_model(feature,"feature_{:.0f}_{:.0f}".format(rmse_train,rmse_test))
            saved+=1
        else:
            notsaved+=1
            print('Not Saved')
    
    print("###########################################################")
    print("Saved: {} %".format(saved))
    print("Not Saved: {} %".format(notsaved))
    
def run_deep(modelo):
    saved = 0
    notsaved = 0
    for i in range(100):
        print("###########################################################")
        print("Modelo:",i)
        epochs = np.random.randint(20,60)
        print("Epochs:",epochs)
        deep = modelo.get_sequential_model(epochs=epochs)
        rmse_train, rmse_test = modelo.get_rmse_sequential()
        print('RMSE train:',rmse_train,'RMSE test',rmse_test)
        if rmse_train < 550 and rmse_test <= 850:
            save_model(deep,"deep_{:.0f}_{:.0f}".format(rmse_train,rmse_test))
            saved+=1
        else:
            notsaved+=1
            print('Not Saved')
    
    print("###########################################################")
    print("Saved: {} %".format(saved))
    print("Not Saved: {} %".format(notsaved))
    
    
if __name__ == '__main__':
    datos_csv = 'Datasets/DataSetVictimasAnios5x1.csv'
    modelo = ModeloSecuestros(csv=datos_csv, random_state = 0)
    
    run_deep(modelo)
    #run_feature(modelo)
    
    #print('###########################################')
    #print(modelo.get_sequential_model().summary())
    #print('Manual MSE',modelo.get_rmse_sequential())
    #print('###########################################')
    #modelo.plot_train()
    
    #print(modelo.X_train_lstm)
    #print(modelo.get_lstm_model().summary())
    #print('LSTM MSE:',modelo.get_rmse_lstm())
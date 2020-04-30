import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
#from sklearn.externals import joblib
from sklearn import svm

from signal import signal, SIGINT
from sys import exit

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

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
    
class ModeloSecuestros:
    def __init__(self, 
                 csv="", 
                 inputs=['Input_{}'.format(i) for i in range(5)],
                 target='Output_0',
                 test_size = 0.09,
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
        self.lstm_trained = False
        self.clf_trained = False
        self.mlp_trained = False
        
        
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
            layers.Dense(300, activation='relu'),
            layers.Dense(300, activation='relu'),
            #layers.Dense(100, activation='relu'),
            #layers.Dense(10, activation='relu'),
            layers.Dense(1, activation='relu')
        ])
        
        self.feature_model.compile(
            optimizer= tf.keras.optimizers.Adam(),
            loss=losses.mean_squared_error,
            metrics=metrics
        )
        
        self.feature_model.fit(self.train_ds,epochs=epochs, verbose=0, shuffle=shuffle)
        self.feature_train_prediction = self.feature_model.predict(self.train_ds)
        self.feature_prediction = self.feature_model.predict(self.test_ds)
        self.feature_train_rmse = rmse(self.y_train,self.feature_train_prediction),
        self.feature_test_rmse  = rmse(self.y_test,self.feature_prediction)
        self.feature_trained = True
        
        return self.feature_model
    
    def get_rmse_feature(self):
        if self.feature_trained:
            return self.feature_train_rmse, self.feature_test_rmse
        else:
            return 'Modelo sin inicializar'
    
    def get_predictions_flat(self, train, test):
        train_prediction_length=len(train)
        train_decode = list(train.reshape((train_prediction_length,)))
        test_prediction_length =len(test)
        test_decode = list(test.reshape((test_prediction_length,)))

        y_values=[]
        y_values.extend(train_decode)
        y_values.extend(test_decode)           
        return y_values
    
    def get_scatter(self,train,test):
        y_values = self.get_predictions_flat(train,test)

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
    
    #########################################################################################################################
    #########################################################################################################################
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
    
    #########################################################################################################################
    #### SVM
    #########################################################################################################################
    def get_svm_model(self):        
        self.clf_model = svm.SVC()
        self.clf_model.fit(self.X_train,self.y_train)
        self.clf_train_prediction = self.clf_model.predict(self.X_train)
        self.clf_prediction = self.clf_model.predict(self.X_test)
        self.clf_train_rmse = rmse(self.y_train,self.clf_train_prediction)
        self.clf_test_rmse  = rmse(self.y_test,self.clf_prediction)
        self.clf_trained =True
        
        return self.clf_model
        
    def get_svm_scatter(self):
        if self.clf_trained:
            self.get_scatter(self.clf_train_prediction,self.clf_prediction);
        else:
            return 'Modelo sin inicializar' 
        
    def get_rmse_svm(self):
        if self.sequential_trained:
            #train,test
            #print(self.sequential_train_prediction)
            return self.clf_train_rmse, self.clf_test_rmse
        else:
            return 'Modelo sin inicializar'

    #########################################################################################################################
    #### MLPREGRESSOR
    #########################################################################################################################
    
    def get_mlp_model(self, hidden_layers=(300,200,100), max_iter=20):
        self.mlp_description = {
            'hidden_layers':hidden_layers,
            'max_iter':max_iter,
            'activation':'relu',
            'solver':'adam'
        }
        self.mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, activation='relu', solver='adam', max_iter=max_iter)
        self.mlp.fit(self.X_train,self.y_train)
        self.mlp_description['n_iter'] = self.mlp.n_iter_
        self.mlp_train_prediction = self.mlp.predict(self.X_train)
        self.mlp_test_prediction = self.mlp.predict(self.X_test)
        self.mlp_train_rmse = rmse(self.y_train,self.mlp_train_prediction)
        self.mlp_test_rmse  = rmse(self.y_test,self.mlp_test_prediction)
        self.mlp_trained=True
        return self.mlp
    
    def get_rmse_mlp(self):
        if self.mlp_trained:
            return self.mlp_train_rmse, self.mlp_test_rmse
        else:
            return 'Modelo sin inicializar'
    
    def get_mlp_scatter(self):
        if self.mlp_trained:
            self.get_scatter(self.mlp_train_prediction,self.mlp_test_prediction);
        else:
            return 'Modelo sin inicializar' 
    #########################################################################################################################
    #########################################################################################################################
    def save_feature_model(self,name):
        self.feature_model.save("Modelos/{}".format(name))
        
    def load_feature_model(self, name):
        self.feature_model = tf.keras.models.load_model('Modelos/{}'.format(name))
        self.feature_train_prediction = self.feature_model.predict(self.train_ds)
        self.feature_prediction = self.feature_model.predict(self.test_ds)
        self.feature_train_rmse = rmse(self.y_train,self.feature_train_prediction),
        self.feature_test_rmse  = rmse(self.y_test,self.feature_prediction)
        self.feature_trained=True
        return self.feature_model
        
    def save_sequential_model(self,name):
        self.sequential_model.save("Modelos/{}".format(name))
        
    def load_sequential_model(self, name):
        self.sequential_model = tf.keras.models.load_model('Modelos/{}'.format(name))
        self.sequential_train_prediction = self.sequential_model.predict(self.X_train)
        self.sequential_prediction = self.sequential_model.predict(self.X_test)
        self.sequential_trained = True
        return self.sequential_model
    
    def save_mlp_model(self,name):
        if self.mlp_trained:
            joblib.dump(self.mlp, 'Modelos/{}.pkl'.format(name)) 
            with open('Modelos/{}_description.json'.format(name), 'w') as file:
                file.write(json.dumps(self.mlp_description))
        else:
            return 'Modelo sin inicializar' 
        
    def load_mlp_model(self,name):
        self.mlp_description = json.loads(open('Modelos/{}_description.json'.format(name)).read())
        self.mlp = joblib.load('Modelos/{}.pkl'.format(name))
        self.mlp_train_prediction = self.mlp.predict(self.X_train)
        self.mlp_test_prediction = self.mlp.predict(self.X_test)
        self.mlp_train_rmse = rmse(self.y_train,self.mlp_train_prediction)
        self.mlp_test_rmse  = rmse(self.y_test,self.mlp_test_prediction)
        self.mlp_trained=True
        return self.mlp
    
    #########################################################################################################################
    #########################################################################################################################
    def plot_in_time(self):
        years = [i for i in range(1982,2021)]
        year_values = []
        year_values.extend(self.X_values[0])
        year_values.extend(list(self.y_values))
        
        self.get_svm_model()
        self.load_mlp_model('mlp_188_735')
        self.load_feature_model('feature_447_691')
        
        #first_years = list(self.X_values[0])
        svm_predictions = list(self.X_values[0])
        mlp_predictions = list(self.X_values[0])
        feature_predictions = list(self.X_values[0])
        svm_predictions.extend(self.get_predictions_flat(self.clf_train_prediction, self.clf_prediction))
        mlp_predictions.extend(self.get_predictions_flat(self.mlp_train_prediction, self.mlp_test_prediction))
        feature_predictions.extend(self.get_predictions_flat(self.feature_train_prediction, self.feature_prediction))
        
        xticks = np.arange(min(years), max(years)+2, 2)
        fig, ax = plt.subplots(2,2, figsize=(30,20))
        
        ax[0,0].plot(years, year_values, 'k-', label="Datos")
        ax[0,1].plot(years, svm_predictions, 'r-', label = "SVM")
        ax[1,0].plot(years, mlp_predictions, 'g-', label = "MLPRegressor")
        ax[1,1].plot(years, feature_predictions, 'b-',label = 'DenseFeatures')
        
        
        for i in range(2):
            for j in range(2):
                ax[i,j].set_xticks(xticks)
                ax[i,j].legend(loc='upper left',fontsize=16)
                ax[i,j].set_xlabel('Años',fontsize=16)
                ax[i,j].set_ylabel('Victimas de Secuestro',fontsize=16)
                plt.setp(ax[i,j].get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=14)
                plt.setp(ax[i,j].get_yticklabels(), fontsize=14)
        
        plt.show()
        fig.savefig('Gráficas/predicción_modelos.png', dpi=300)
        
        #return fig
    
    def df_feature_from_list(self,values):
        feature = {}
        for i,j in zip(self.inputs,values):
            feature[i] = [j]
        
        feature['Output_0'] = [0]
        
        #print(feature)
        
        df_feature = pd.DataFrame.from_dict(feature)
        X = df_feature[self.inputs]
        y = df_feature['Output_0'].values
        ds_feature = tf.data.Dataset.from_tensor_slices((dict(X),y))#df_to_dataset(df_feature)
        ds_feature = ds_feature.batch(1)
        
        return ds_feature
        
    def future(self, future_year=2031):
        len_inputs = len(self.inputs)
        years = [i for i in range(2016,future_year)]
        
        start_years = list(self.y_values)[-len_inputs:]
        
        svm = self.get_svm_model()
        mlp = self.load_mlp_model('mlp_188_735')
        feature = self.load_feature_model('feature_447_691')
        
        svm_future = start_years[:]
        mlp_future = start_years[:]
        feature_future = start_years[:]
        
        for i in range(future_year-2021):
            test_ds = self.df_feature_from_list(feature_future[-len_inputs:])
            
            svm_next_year = svm.predict([svm_future[-len_inputs:]])
            mlp_next_year = mlp.predict([mlp_future[-len_inputs:]])
            feature_next_year = feature.predict(test_ds)
            
            svm_future.append(svm_next_year[0])
            mlp_future.append(mlp_next_year[0])
            feature_future.append(feature_next_year[0][0])
            
            #print(i,svm_next_year, mlp_next_year, feature_next_year[0][0])

        #print('Datos SVM 2016-2030', svm_future)
        #print('Datos MLP 2016-2030', mlp_future)
        #print('Datos Features 2016-2030', feature_future)
        
        max_svm = max(svm_future)
        max_mlp = max(mlp_future)
        max_feat = max(feature_future)
        max_y = max(max_svm,max_mlp,max_feat)
        
        min_svm = min(svm_future)
        min_mlp = min(mlp_future)
        min_feat = min(feature_future)
        min_y = min(min_svm,min_mlp,min_mlp)
        
        xticks = np.arange(min(years), max(years)+1, 1)
        yticks = np.arange(min_y, max_y+1, 500)
        
        fig, ax = plt.subplots(3,1,figsize=(10,20))
        
        ax[0].plot(years, svm_future, 'r-', label = "SVM")
        ax[1].plot(years, mlp_future, 'g-', label = "MLPRegressor")    
        ax[2].plot(years, feature_future, 'b-',label = 'DenseFeatures')
        
        for i in range(3):
            ax[i].set_xticks(xticks)
            ax[i].set_yticks(yticks)
            ax[i].legend(loc='upper left',fontsize=16)
            ax[i].set_xlabel('Años',fontsize=16)
            ax[i].set_ylabel('Victimas de Secuestro',fontsize=16)
            plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=14)
            plt.setp(ax[i].get_yticklabels(), fontsize=14)
        
        plt.show()
        fig.savefig('Gráficas/futuro_2030.png', dpi=300)
            
        
def run(model, model_type,train_th=800, test_th=560, itr=100, break_on_save=False, inf=False):
    saved = 0
    notsaved = 0
    i = 0
    while i < itr or inf:
        print("###########################################################")
        print("Modelo:",i)
        epochs = np.random.randint(20,40)
        get_model = getattr(model, 'get_{}_model'.format(model_type))
        the_model = get_model(epochs=epochs,shuffle=False)
        
        get_rmse  = getattr(model, 'get_rmse_{}'.format(model_type))
        rmse_train, rmse_test = get_rmse()
        
        print('RMSE train:',rmse_train,'RMSE test',rmse_test)
        if rmse_train <= train_th and rmse_test <= test_th:
            save_model = getattr(model, 'save_{}_model'.format(model_type))
            save_model("{}_{:.0f}_{:.0f}".format(model_type,rmse_train,rmse_test))
            saved+=1
            if break_on_save:
                break
        else:
            notsaved+=1
            print('Not Saved')
        
        i+=1
    
    print("###########################################################")
    print("Saved: {:.2f} %".format((saved*100)/saved+notsaved))
    print("Not Saved: {:.2f} %".format((notsaved*100)/saved+notsaved))

def run_mlp():
    for i in range(100):
        hidden_layers = tuple([np.random.randint(100, 300) for _ in range(np.random.randint(1,5))])
        max_iter = np.random.randint(20,200)
        modelo.get_mlp_model(hidden_layers=hidden_layers, max_iter=max_iter)
        rmse_train, rmse_test = modelo.get_rmse_mlp()
        print('RMSE train:',rmse_train,'RMSE test',rmse_test)
        if rmse_train <= 200 and rmse_test <= 770:
            modelo.save_mlp_model("mlp_{:.0f}_{:.0f}".format(rmse_train,rmse_test))
            print('Saved MLP')
            break
                    
if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    print('Running. Press CTRL-C to exit.')
    datos_csv = 'Datasets/DataSetVictimasAnios5x1.csv'
    modelo = ModeloSecuestros(csv=datos_csv, random_state = 1)
    
    #run_mlp()    
    modelo.future()
    #run(modelo,'feature', train_th=450,test_th=700, inf=True)
    #modelo.get_mlp_model()
    #print(modelo.get_rmse_mlp())
    #run_deep(modelo)
    #run_feature(modelo, test_th=750,break_on_save=True, inf=True)
    #modelo.load_feature_model('feature2_1306_641')
    #loaded.load_weights('feature_526.h5')
    
    #modelo.load_feature_model('Modelos/feature_526')
    #print(modelo.get_rmse_feature())
    #print(getattr(modelo,'get_rmse_feature')())
    #print('###########################################')
    #print(modelo.get_sequential_model().summary())
    #print('Manual MSE',modelo.get_rmse_sequential())
    #print('###########################################')
    #modelo.plot_train()
    
    #print(modelo.X_train_lstm)
    #print(modelo.get_lstm_model().summary())
    #print('LSTM MSE:',modelo.get_rmse_lstm())

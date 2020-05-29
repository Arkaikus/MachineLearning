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
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
#from sklearn.externals import joblib
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
    
from signal import signal, SIGINT
from sys import exit

def handler(signal_received, frame):
    '''Exit with CTRL+C'''
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(X,y, shuffle=False, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds

def train_test_ds(X,y, test_size=0.3, batch_size=5, random_state=1, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state = random_state, shuffle=shuffle)
    train_ds = df_to_dataset(X_train,y_train, batch_size=batch_size)
    test_ds = df_to_dataset(X_test,y_test, shuffle=False, batch_size=batch_size)
    
    return (train_ds,test_ds)

def rmse(y_train,y_pred):
    return np.sqrt(mean_squared_error(y_train,y_pred))

class Framework:
    def __init__(self, 
                 dataframe,
                 start_year,
                 end_year,
                 inputs=['Input_{}'.format(i) for i in range(5)],
                 target='Output_0',
                 test_size = 0.09,
                 random_state = 1,
                 save_dir = 'Models',
                 shuffleDataset = False):
        
        if not isinstance(dataframe, pd.DataFrame):
            raise Exception('Not a Pandas Dataframe')
            
        #saving dataframe
        self.df = dataframe.copy()
        
        #data start year
        self.start_year = start_year
        #data latest year
        self.end_year = end_year
        
        #input columns
        self.inputs = inputs
        #input len
        self.len_inputs = len(inputs)
        #target column
        self.target = target
        #test set size
        self.test_size = test_size
        #traint set random state
        self.random_state = random_state
        #save folder
        self.save_dir = save_dir
        #inputs dataframe
        self.X = self.df[inputs]
        #inputs values
        self.X_values = self.X.values
        #output dataframe
        self.y = self.df[target]
        #output values
        self.y_values = self.y.values
        
        #############################################################
        # Keras Batch Dataset
        self.train_ds, self.test_ds = train_test_ds(self.X,self.y, test_size=test_size, random_state=random_state, shuffle=shuffleDataset)
        
        #############################################################
        # SKLearn Dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_values, 
                                                                                self.y_values, 
                                                                                test_size = test_size, 
                                                                                random_state = random_state, 
                                                                                shuffle=shuffleDataset)
        
        #############################################################
        # For every model if it's trained
        self.feature_trained = False
        self.sequential_trained = False
        self.svm_trained = False
        self.mlp_trained = False
        
    #########################################################################################################################
    ## UTILS
    #########################################################################################################################
    
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
    
    def get_predictions_flat(self, train, test):
        train_prediction_length=len(train)
        train_decode = list(train.reshape((train_prediction_length,)))
        test_prediction_length =len(test)
        test_decode = list(test.reshape((test_prediction_length,)))

        y_values=[]
        y_values.extend(train_decode)
        y_values.extend(test_decode)           
        return y_values
    
    def get_scatter(self,train,test, title="Dispersión", 
                    figsize = (8,8),
                    color='',
                    edgecolor=(0,0,1,1),
                    title_fontsize = 28,
                    label_fontsize = 24,
                    legend_fontsize = 22,
                    ticks_fontsize = 18):
        
        y_values = self.get_predictions_flat(train,test)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.y_values, self.y_values, 'k-')
        ax.plot(self.y_values, y_values, 'D'+color,
                markerfacecolor='w',
                markeredgewidth=1.5, 
                markeredgecolor=edgecolor)
        ax.set_xlabel('Valores Esperados', fontsize=label_fontsize)
        ax.set_ylabel('Predicciones', fontsize=label_fontsize)
        ax.set_title(title,fontsize=title_fontsize)
        plt.setp(ax.get_xticklabels(), rotation=30,  horizontalalignment='right', fontsize=ticks_fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=ticks_fontsize)
        
        plt.tight_layout()
        return fig
    
    def read_json(self,name):
        return json.loads(open('{}/{}'.format(self.save_dir,name)).read())

    def write_json(self,name, a_dict):
        with open('{}/{}'.format(self.save_dir,name), 'w') as file:
            file.write(json.dumps(a_dict))
    #########################################################################################################################
    ## DEEP FEATURES MODEL
    #########################################################################################################################
    def get_feature_model(self,hidden_layers=(300,200,100), epochs = 20, shuffle=False):
        self.feature_model_description = {
            'inputs':'keras.layers.DenseFeatures',
            'hidden_layers':hidden_layers,
            'epochs':epochs,
            'activation':'relu',
            'solver':'keras.optimizers.Adam'
        }
        
        feature_columns = []
        for header in self.inputs:
            feature_columns.append(feature_column.numeric_column(header))
        
        feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
        self.feature_model = tf.keras.Sequential()
        self.feature_model.add(feature_layer)
        
        for units in hidden_layers:
            self.feature_model.add(layers.Dense(units,activation='relu'))
            
        self.feature_model.add(layers.Dense(1, activation='relu'))
        self.feature_model.compile(
            optimizer= tf.keras.optimizers.Adam(),
            loss=losses.mean_squared_error,
            metrics=['mean_squared_error']
        )
        
        self.feature_model.fit(self.train_ds,epochs=epochs, verbose=0, shuffle=shuffle)
        self.do_feature_predictions()
        self.feature_trained = True
        
        return self.feature_model
    
    def do_feature_predictions(self):
        self.feature_train_prediction = self.feature_model.predict(self.train_ds)
        self.feature_test_prediction  = self.feature_model.predict(self.test_ds)
        self.feature_predictions_flat = self.get_predictions_flat(self.feature_train_prediction, self.feature_test_prediction)
        self.feature_train_rmse = rmse(self.y_train,self.feature_train_prediction)
        self.feature_test_rmse  = rmse(self.y_test,self.feature_test_prediction)
        
    def get_feature_rmse(self):
        if self.feature_trained:
            return self.feature_train_rmse, self.feature_test_rmse
        else:
            return 'Modelo sin inicializar'
    
    def get_feature_scatter(self):
        if self.feature_trained:
            fig = self.get_scatter(self.feature_train_prediction,self.feature_test_prediction, title="Dispersión Red Neuronal Keras")
            fig.savefig("{}/feature_scatter.png".format(self.save_dir), dpi=300)
        else:
            return 'Modelo sin inicializar'
    
    def get_feature_predictions(self):
        if self.feature_trained:
            return self.feature_predictions_flat
        else:
            return 'Modelo sin inicializar'

    #########################################################################################################################
    ## DEEP MODEL
    #########################################################################################################################
    def get_sequential_model(self, hidden_layers=tuple(),epochs = 20,shuffle=False):
        self.sequential_model_description = {
            'inputs':'keras.layers.Dense',
            'hidden_layers':hidden_layers,
            'epochs':epochs,
            'activation':'relu',
            'solver':'keras.optimizers.Adam'
        }
        
        self.sequential_model = tf.keras.Sequential()
        self.sequential_model.add(layers.Dense(self.len_inputs, input_dim=self.len_inputs))

        for units in hidden_layers:
            self.sequential_model.add(layers.Dense(units,activation='relu'))
        
        self.sequential_model.add(layers.Dense(1))
        
        self.sequential_model.compile(loss=losses.mean_squared_error , optimizer="adam", metrics=['mean_squared_error'])
        self.sequential_model.fit(x=self.X_train, y=self.y_train, epochs=epochs, verbose=0, shuffle=shuffle)
        self.do_sequential_predictions()
        self.sequential_trained = True
        
        return self.sequential_model

    def do_sequential_predictions(self):
        self.sequential_train_prediction = self.sequential_model.predict(self.X_train)
        self.sequential_test_prediction  = self.sequential_model.predict(self.X_test)
        self.sequential_predictions_flat = self.get_predictions_flat(self.sequential_train_prediction, self.sequential_test_prediction)
        self.sequential_train_rmse = rmse(self.y_train,self.sequential_train_prediction)
        self.sequential_test_rmse  = rmse(self.y_test,self.sequential_test_prediction)
        
    def get_sequential_rmse(self):
        if self.sequential_trained:
            #train,test
            #print(self.sequential_train_prediction)
            return self.sequential_train_rmse, self.sequential_test_rmse
        else:
            return 'Modelo sin inicializar'
    
    def get_sequential_scatter(self,figsize=(8,8),
                               title_fontsize = 20,
                               label_fontsize = 18,
                               ticks_fontsize = 18):
        if self.sequential_trained:
            self.sequential_scatter_fig = self.get_scatter(self.sequential_train_prediction,self.sequential_test_prediction, 
                                                           title="Dispersión Red Neuronal Keras",figsize=figsize,color="b",
                                                          title_fontsize=title_fontsize,
                                                          label_fontsize=label_fontsize,
                                                          ticks_fontsize=ticks_fontsize)
            self.sequential_scatter_fig.savefig("{}/sequential_scatter.png".format(self.save_dir), dpi=300)
        else:
            return 'Modelo sin inicializar'
        
    def get_sequential_predictions(self):
        if self.sequential_trained:
            return self.sequential_predictions_flat
        else:
            return 'Modelo sin inicializar'
    
    #########################################################################################################################
    #### SVM
    #########################################################################################################################
    def get_svm_model(self):
        #self.svm_model = make_pipeline(StandardScaler(), svm.LinearSVR(random_state=0, tol=1e-5))
        self.svm_model = svm.SVR(kernel="linear")
        self.svm_model.fit(self.X_train,self.y_train)
        self.do_svm_predictions()
        self.svm_trained =True
        
        return self.svm_model
    
    def do_svm_predictions(self):
        self.svm_train_prediction = self.svm_model.predict(self.X_train)
        self.svm_test_prediction  = self.svm_model.predict(self.X_test)
        self.svm_predictions_flat = self.get_predictions_flat(self.svm_train_prediction, self.svm_test_prediction)
        self.svm_train_rmse = rmse(self.y_train,self.svm_train_prediction)
        self.svm_test_rmse  = rmse(self.y_test ,self.svm_test_prediction)
        
    def get_svm_scatter(self,figsize=(8,8),
                        title_fontsize = 20,
                        label_fontsize = 18,
                        ticks_fontsize = 18):
        if self.svm_trained:
            self.svm_scatter_fig = self.get_scatter(self.svm_train_prediction,self.svm_test_prediction, title="Dispersión Máquina de Soporte Vectorial",
                                                    figsize=figsize, color="r", edgecolor=(1,0,0,1),
                                                    title_fontsize=title_fontsize,
                                                    label_fontsize=label_fontsize,
                                                    ticks_fontsize=ticks_fontsize)
            
            self.svm_scatter_fig.savefig("{}/svm_scatter.png".format(self.save_dir), dpi=300)
        else:
            return 'Modelo sin inicializar' 
        
    def get_svm_rmse(self):
        if self.svm_trained:
            return self.svm_train_rmse, self.svm_test_rmse
        else:
            return 'Modelo sin inicializar'
    
    def get_svm_predictions(self):
        if self.svm_trained:
            return self.svm_predictions_flat[:]
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
        self.do_mlp_predictions()
        self.mlp_trained=True
        return self.mlp
    
    def do_mlp_predictions(self):
        self.mlp_train_prediction = self.mlp.predict(self.X_train)
        self.mlp_test_prediction  = self.mlp.predict(self.X_test)
        self.mlp_predictions_flat = self.get_predictions_flat(self.mlp_train_prediction, self.mlp_test_prediction)
        self.mlp_train_rmse = rmse(self.y_train,self.mlp_train_prediction)
        self.mlp_test_rmse  = rmse(self.y_test,self.mlp_test_prediction)
        
    def get_mlp_rmse(self):
        if self.mlp_trained:
            return self.mlp_train_rmse, self.mlp_test_rmse
        else:
            return 'Modelo sin inicializar'
    
    def get_mlp_scatter(self,figsize=(8,8),
                        title_fontsize = 20,
                        label_fontsize = 18,
                        ticks_fontsize = 18):
        if self.mlp_trained:
            self.mlp_scatter_fig = self.get_scatter(self.mlp_train_prediction,self.mlp_test_prediction, title="Dispersión Red Neuronal MLPRegressor",
                                                    figsize=figsize, color="r", edgecolor=(0,1,0,1),
                                                    title_fontsize=title_fontsize,
                                                    label_fontsize=label_fontsize,
                                                    ticks_fontsize=ticks_fontsize)
            self.mlp_scatter_fig.savefig("{}/mlp_scatter.png".format(self.save_dir), dpi=300)
        else:
            return 'Modelo sin inicializar' 
    
    def get_mlp_predictions(self):
        if self.mlp_trained:
            return self.mlp_predictions_flat[:]
        else:
            return 'Modelo sin inicializar' 
    
    #########################################################################################################################
    ## SAVING AND LOADING
    #########################################################################################################################
    def save_feature_model(self,name):
        self.feature_model.save("{}/{}".format(self.save_dir,name))
        self.write_json('{}/description.json'.format(name), self.feature_model_description)
        
    def load_feature_model(self, name):
        self.feature_model = tf.keras.models.load_model('{}/{}'.format(self.save_dir,name))
        self.feature_model_description = self.read_json('{}/description.json'.format(name))
        self.do_feature_predictions()
        self.feature_trained=True
        return self.feature_model
        
    def save_sequential_model(self,name):
        self.sequential_model.save("{}/{}".format(self.save_dir,name))
        self.write_json('{}/description.json'.format(name), self.sequential_model_description)
        
    def load_sequential_model(self, name):
        self.sequential_model = tf.keras.models.load_model('{}/{}'.format(self.save_dir,name))
        self.sequential_model_description = self.read_json('{}/description.json'.format(name))
        self.do_sequential_predictions()
        self.sequential_trained = True
        return self.sequential_model
    
    def save_mlp_model(self,name):
        if self.mlp_trained:
            joblib.dump(self.mlp, '{}/{}.pkl'.format(self.save_dir,name))
            self.write_json('{}_description.json'.format(name), self.mlp_description)
        else:
            return 'Modelo sin inicializar'
        
    def load_mlp_model(self,name):
        self.mlp_description = self.read_json('{}_description.json'.format(name))
        self.mlp = joblib.load('{}/{}.pkl'.format(self.save_dir,name))
        self.do_mlp_predictions()
        self.mlp_trained=True
        return self.mlp
    
    #########################################################################################################################
    
    def plot_model(self,model):
        return plot_model(model,show_shapes=True, to_file='{}/model.png'.format(self.save_dir), rankdir='LR')
    
    #########################################################################################################################
    
    def df_feature_from_list(self,values):
        feature = {}
        for i,j in zip(self.inputs,values):
            feature[i] = [j]
        
        feature[self.target] = [0]
        
        df_feature = pd.DataFrame.from_dict(feature)
        X = df_feature[self.inputs]
        y = df_feature['Output_0'].values
        ds_feature = tf.data.Dataset.from_tensor_slices((dict(X),y))#df_to_dataset(df_feature)
        ds_feature = ds_feature.batch(1)
        
        return ds_feature
    
    def plot_compare(self, 
                     plot_save_location,
                     x_label="",
                     y_label="",
                     figsize=(18,50),
                     dpi=120,
                     title_fontsize = 20,
                     label_fontsize = 16,
                     legend_fontsize = 16,
                     ticks_fontsize = 14,
                     xticks_div = 3,
                     yticks_div = 6
                    ):
        
        years = list(range(self.start_year,self.end_year+1))
        years4models = years[self.len_inputs:]
        ## ORIGINAL DATA
        year_values = []
        year_values.extend(self.X_values[0])
        year_values.extend(list(self.y_values))
        #print(year_values)
        
        if not(self.mlp_trained and self.sequential_trained and self.svm_trained):
            raise Exception('Modelos sin inicializar')
            return None
        
        svm_predictions = self.get_svm_predictions()
        mlp_predictions = self.get_mlp_predictions()
        sequential_predictions = self.get_sequential_predictions()
        
        ## PLOT
        #print("Max Year",max(years))
        #step = (2020 - min(years))//xticks_div
        #print("X_step",step)
        #step = step if step%2 == 0 else step-1
        xticks = [i for i in range(min(years),max(years),5)]#np.arange(min(years), max(years)+step, step)
        xticks += [max(years)]
        xticks = np.unique(xticks)
        max_svm = max(svm_predictions)
        max_mlp = max(mlp_predictions)
        max_seq = max(sequential_predictions)
        max_y = max(max_svm,max_mlp,max_seq)
        
        min_svm = min(svm_predictions)
        min_mlp = min(mlp_predictions)
        min_seq = min(sequential_predictions)
        min_y = min(min_svm,min_mlp,min_seq)
        
        step = (max_y-min_y)//yticks_div
        #print("Y_step",step)
        yticks = np.arange(min_y, max_y, step)
        
        fig, ax = plt.subplots(4,1, figsize=figsize)
        #ax[2,1].delaxes()
        
        ax[0].set_title('Datos Originales', fontsize=title_fontsize)
        for i in range(4):
            ax[i].plot(years, year_values, 'Dk--', label="Datos", markerfacecolor='w',markeredgewidth=1.5)
            sns.regplot(x=years, y=year_values, ax=ax[i], label="Regresión", scatter=False, ci=0)
        
        
        ax[1].set_title('Máquina de Soporte Vectorial (SVR)', fontsize=title_fontsize)
        ax[1].plot(years4models, svm_predictions, 'Dr-', label = "SVM", 
                     markerfacecolor='w',
                     markeredgewidth=1.5, 
                     markeredgecolor=(1, 0, 0, 1))
        
        

        ax[2].set_title('Red Neuronal MLPRegressor', fontsize=title_fontsize)
        ax[2].plot(years4models, mlp_predictions, 'Dg-', label = "MLPRegressor",
                     markerfacecolor='w',
                     markeredgewidth=1.5, 
                     markeredgecolor=(0, 1, 0, 1))
        
        ax[3].set_title('Red Neuronal Keras', fontsize=title_fontsize)
        ax[3].plot(years4models, sequential_predictions, 'Db-', label = 'Keras',
                     markerfacecolor='w',
                     markeredgewidth=1.5, 
                     markeredgecolor=(0, 0, 1, 1))
        
        #fig.delaxes(ax[2,1])
        
        for i in range(4):
            ax[i].set_xticks(xticks)
            ax[i].set_yticks(yticks)
            ax[i].legend(loc='upper left',fontsize=legend_fontsize)
            ax[i].set_xlabel(x_label,fontsize=label_fontsize)
            ax[i].set_ylabel(y_label,fontsize=label_fontsize)
            plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=ticks_fontsize)
            plt.setp(ax[i].get_yticklabels(), fontsize=ticks_fontsize)
        
        plt.tight_layout()
        plt.show()
        fig.savefig(plot_save_location, dpi=dpi)
        
    def plot_future(self,
                    plot_save_location,
                    end_year,
                    x_label="",
                    y_label="",
                    figsize=(18,30),
                    dpi=120,
                    title_fontsize = 20,
                    label_fontsize = 16,
                    legend_fontsize = 16,
                    ticks_fontsize = 14,
                    start_offset=28,
                    xticks_div = 3,
                    yticks_div = 6
                   ):
        
        #if self.end_year-start_year < self.len_inputs and start_year-self.start_year < self.len_inputs:
        #    raise Exception('start_year must be at least {} and at most{}'.format(self.start_year+self.len_inputs,self.end_year-self.len_inputs))
        
        start_year = self.start_year+self.len_inputs
        
        #normalization
        start_offset = start_offset if start_year+start_offset<=self.end_year-self.len_inputs+1 else self.end_year-start_year-self.len_inputs+1
        start_year += start_offset
        
        years = list(range(start_year,end_year+1))
        y_values = list(self.y_values)
        start_years = y_values[start_offset:start_offset+self.len_inputs]
        
        data_values = y_values[start_offset:]
        data_years = years[:len(data_values)]
        print(data_years)
        #svm = self.get_svm_model()
        
        if not(self.mlp_trained and self.sequential_trained and self.svm_trained):
            raise Exception('Modelos sin inicializar')
            return None
    
        svm_future = start_years[:]
        mlp_future = start_years[:]
        sequential_future = start_years[:]
        
        for i in range(end_year-start_year-self.len_inputs+1):
            #print(feature_future[-self.len_inputs:])
            #test_ds = self.df_feature_from_list(feature_future[-self.len_inputs:])
            
            svm_next_year = self.svm_model.predict([svm_future[-self.len_inputs:]])
            mlp_next_year = self.mlp.predict([mlp_future[-self.len_inputs:]])
            #feature_next_year = feature.predict(test_ds)
            
            sequential_future_predict = np.array(sequential_future[-self.len_inputs:]).reshape((1,5))
            sequential_next_year = self.sequential_model.predict(sequential_future_predict)
        
            svm_future.append(svm_next_year[0])
            mlp_future.append(mlp_next_year[0])
            sequential_future.append(sequential_next_year[0][0])
        
        max_svm = max(svm_future)
        max_mlp = max(mlp_future)
        #max_feat = max(feature_future)
        max_seq = max(sequential_future)
        max_y = max(max_svm,max_mlp,
                    #max_feat,
                    max_seq)
        
        min_svm = min(svm_future)
        min_mlp = min(mlp_future)
        #min_feat = min(feature_future)
        min_seq = min(sequential_future)
        min_data = min(data_values)
        min_y = min(min_svm,min_mlp,
                    #min_feat,
                    min_seq, min_data)
        
        step = (end_year - min(years))//xticks_div
        xticks = np.arange(min(years), max(years)+step, step)
        
        step = (max_y-min_y)//yticks_div
        yticks = np.arange(min_y, max_y+step, step)
        
        fig, ax = plt.subplots(3,1,figsize=figsize)
        
        ax[0].set_title("Predicción Máquina de Soporte Vectorial al Año {}".format(end_year), fontsize=title_fontsize)
        ax[0].plot(years[self.len_inputs-1:], svm_future[self.len_inputs-1:], 'Dr-', label = "SVM",
                   markerfacecolor='w',
                   markeredgewidth=1.5, 
                   markeredgecolor=(1, 0, 0, 1))
        #ax[0].plot(years[:self.len_inputs], svm_future[:self.len_inputs], 'Dk--', label = "Datos", markerfacecolor='w')
        
        ax[1].set_title("Predicción Red Neuronal MLPRegressor al Año {}".format(end_year), fontsize=title_fontsize)
        ax[1].plot(years[self.len_inputs-1:], mlp_future[self.len_inputs-1:], 'Dg-', label = "MLPRegressor",
                   markerfacecolor='w',
                   markeredgewidth=1.5, 
                   markeredgecolor=(0, 1, 0, 1))    
        #ax[1].plot(years[:self.len_inputs], mlp_future[:self.len_inputs], 'Dk--', label = "Datos", markerfacecolor='w')
        #ax[2].plot(years, feature_future, 'b-',label = 'DenseFeatures')
        
        ax[2].set_title("Predicción Red Neuronal Keras al Año {}".format(end_year), fontsize=title_fontsize)
        ax[2].plot(years[self.len_inputs-1:], sequential_future[self.len_inputs-1:], 'Db-',label = 'Keras',
                   markerfacecolor='w',
                   markeredgewidth=1.5, 
                   markeredgecolor=(0, 0, 1, 1))
        #ax[2].plot(years[:self.len_inputs], sequential_future[:self.len_inputs], 'Dk--', label = "Datos", markerfacecolor='w')
        
        for i in range(3):
            ax[i].plot(data_years, data_values, 'Dk--', label="Datos",markerfacecolor='w',markeredgewidth=1.5)
            ax[i].set_xticks(xticks)
            ax[i].set_yticks(yticks)
            ax[i].legend(loc='upper left',fontsize=legend_fontsize)
            ax[i].set_xlabel(x_label,fontsize=label_fontsize)
            ax[i].set_ylabel(y_label,fontsize=label_fontsize)
            plt.setp(ax[i].get_xticklabels(), rotation=30, horizontalalignment='right', fontsize=ticks_fontsize)
            plt.setp(ax[i].get_yticklabels(), fontsize=ticks_fontsize)
        
        plt.tight_layout()
        plt.show()
        fig.savefig(plot_save_location, dpi=dpi)
            

#########################################################################################################################
#########################################################################################################################         
            
def run(model, 
        model_type,
        train_th = 800, 
        test_th  = 560, 
        itr      = 100, 
        inf      = False, 
        epochs   = 20,
        shuffle  = True,
        hidden_layers = tuple(),
        break_on_save = False):
    
    saved = 0
    notsaved = 0
    best=None
    best_description=None
    best_train_rmse=np.inf
    best_test_rmse =np.inf
    
    i = 0
    while i < itr or inf:
        print("###########################################################")
        print("Modelo:",i)
        #epochs = np.random.randint(20,40)
        #hidden_layers = descendant_layers()
        get_model = getattr(model, 'get_{}_model'.format(model_type))
        the_model = get_model(hidden_layers=hidden_layers,epochs=epochs,shuffle=shuffle)
        
        description = getattr(model, '{}_model_description'.format(model_type))
        get_rmse  = getattr(model, 'get_{}_rmse'.format(model_type))
        rmse_train, rmse_test = get_rmse()
        
        if rmse_test<best_test_rmse:
            best_train_rmse = rmse_train
            best_test_rmse = rmse_test
            best = the_model
            best_description = description
        
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
    
    print("Saving best test rmse",)
    name = "{}_{:.0f}_{:.0f}".format(model_type,best_train_rmse,best_test_rmse)
    print(name)
    best.save("{}/{}".format(fw.save_dir,name))
    fw.write_json('{}/description.json'.format(name), best_description)
    
    print("###########################################################")
    print("Saved: {:.2f} %".format((saved*100)/(saved+notsaved)))
    print("Not Saved: {:.2f} %".format((notsaved*100)/(saved+notsaved)))

def descendant_layers():
    nol = np.random.randint(1,5)
    hidden_layers = []
    last_hl=300
    for h in range(nol):
        layer_size = np.random.randint(max(10,last_hl-200), last_hl)
        last_hl = layer_size
        hidden_layers.append(layer_size)
    
    return tuple(hidden_layers)

def run_mlp(fw, hidden_layers=tuple(), max_iter=150,load=False, name=None, overtrain=False):
    best=None
    best_description=None
    best_train_rmse=np.inf
    best_test_rmse =np.inf
    
    if load and name is not None:
        mlp = fw.load_mlp_model(name)
    
    for i in range(100):
        print("###########################################################")
        print("Modelo",i)
        if not load:
            #hidden_layers = hidden_layers#descendant_layers()
            #max_iter = np.random.randint(20,200)
            mlp = fw.get_mlp_model(hidden_layers=hidden_layers, max_iter=max_iter)
        else:
            fw.mlp.fit(fw.X_train,fw.y_train)
            fw.do_mlp_predictions()
            if overtrain and fw.mlp.max_iter <= 700:
                fw.mlp.max_iter += 10
            
        mlp_description = fw.mlp_description
        rmse_train, rmse_test = fw.get_mlp_rmse()
        print('RMSE train:',rmse_train,'RMSE test',rmse_test)
        
        if rmse_test<best_test_rmse:
            best_train_rmse = rmse_train
            best_test_rmse = rmse_test
            best = mlp
            best_description = mlp_description
        
        if rmse_train <= 200 and rmse_test <= 770 and not overtrain:
            name="mlp_{:.0f}_{:.0f}".format(rmse_train,rmse_test)
            fw.save_mlp_model(name)
            print('Saved MLP',name)
            break
    
    name = "mlp_{:.0f}_{:.0f}".format(best_train_rmse,best_test_rmse)
    joblib.dump(best, "{}/{}.pkl".format(fw.save_dir,name))
    fw.write_json('{}_description.json'.format(name), mlp_description)

    
#########################################################################################################################
#########################################################################################################################

if __name__ == '__main__':
    # Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    print('Running. Press CTRL-C to exit.')
    datos_csv = 'Secuestros/Datasets/DataSetVictimasAnios5x1.csv'
    df = pd.read_csv(datos_csv)
    fw = Framework(
        dataframe  = df,
        start_year = 1976,
        end_year   = 2020,
        inputs     = ['Input_{}'.format(i) for i in range(5)],
        target     = 'Output_0',
        save_dir   = "Modelos",
        test_size  = 0.15,
        #shuffleDataset = True
    )
    
    #run_mlp(fw, load=True, name="mlp_164_766", overtrain=True)
    
    #fw.load_mlp_model('mlp_164_766')
    #hidden_layers = (163,130,120,117)#tuple(fw.mlp_description["hidden_layers"])
    hidden_layers = (5,200,150,100,50)
    max_iter = 150 #fw.mlp_description["max_iter"]
    #run(fw,'sequential',hidden_layers=hidden_layers, epochs=max_iter, shuffle=False, break_on_save = True)
    #run_mlp(fw, hidden_layers, max_iter)
    
    #run_mlp(fw, hidden_layers = hidden_layers, max_iter=max_iter)
    
    
    mlp_model = "mlp_181_593"
    sequential_model = "sequential_149_555"

    fw.get_svm_model()
    #fw.load_mlp_model(mlp_model)
    #fw.load_sequential_model(sequential_model)
    #_samples, n_features = df.shape
    #ng = np.random.RandomState(1)
    # = rng.randn(n_samples)
    # = rng.randn(n_samples, n_features)
    #egr = make_pipeline(StandardScaler(), svm.SVR(C=1.0, epsilon=0.2))

    regr = svm.SVR(kernel="linear")
    #regr = make_pipeline(StandardScaler(), svm.LinearSVR(random_state=1, tol=1e-5))
    regr.fit(fw.X_train, fw.y_train)
    
    #svm_model = svm.SVC()
    #svm_model.fit(fw.X_train,fw.y_train)
    
    #print("SVM RMSE:",fw.get_svm_rmse())
    
    #print("SVM TRAIN Y-VALUES PREDICTED", svm_model.predict(fw.X_train))
    regr_train_predict = regr.predict(fw.X_train)
    
    print("SVM TRAIN Y-VALUES PREDICTED", regr_train_predict)
    
    print("SVM TRAIN Y-VALUES DATA", fw.y_train)
    
    print("RMSE TRAIN", rmse(fw.y_train, regr_train_predict))
    
    regr_test_predict = regr.predict(fw.X_test)
    
    print("SVM TEST Y-VALUES PREDICTED", regr_test_predict)
    
    print("SVM TEST Y-VALUES DATA", fw.y_test)
    
    print("RMSE TEST", rmse(fw.y_test, regr_test_predict))
    
    
    #print("MLP RMSE:",fw.get_mlp_rmse())
    #print("Sequential RMSE:",fw.get_sequential_rmse())

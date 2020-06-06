import codecs
import pandas as pd
import numpy as np
import json
import sys
from copy import deepcopy
np.set_printoptions(threshold=sys.maxsize)

def generarDataset(nombre="Enlaces/lista-temporal-50-reverse.csv", nombreSalida="Datasets/dataset50_3.csv", offset=6):
    df = pd.read_csv(nombre)
    vertices = df['nodes'].values
    vertices = list(vertices)
    nfilas = len(vertices)
    dataset = []
    for i in range(nfilas-offset+1):
        dataset.append(vertices[i:i+offset])
        
    df = pd.DataFrame(dataset)
    df.to_csv(nombreSalida, index=False)


def location_dataset(csv_file="Enlaces/lista-temporal-50-reverse-min.csv", offset=6, dim=38):
    df = pd.read_csv(csv_file)
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    g = df[['x','y']].groupby(df['time'].dt.strftime('%Y-%m'))
    matrixes_by_month = np.zeros((g.ngroups, dim, dim))
    group_index = 0
    for key,group in g:
        for x,y in group.itertuples(index=False):
            matrixes_by_month[group_index][y-1][x-1] += 1
        group_index += 1

    x_matrices = np.zeros(((g.ngroups - offset + 1), dim, dim, offset-1))
    y_matrices = np.zeros(((g.ngroups - offset + 1), dim, dim, 1))
    for i in range(g.ngroups - offset + 1):
        x_matrices[i] = matrixes_by_month[i:i + offset - 1].copy().reshape((dim, dim, offset-1))
        y_matrices[i] = matrixes_by_month[i+offset - 1:i + offset].copy().reshape((dim,dim,1))

    return x_matrices,y_matrices


if __name__ == '__main__':
    #generarDataset()
    #location_dataset()
    file= open("Enlaces/lista-temporal-1000.csv",'r')
    file2=open("Enlaces/lista-temporal-50-reverse.csv",'r')
    file3=open("Enlaces/lista-temporal-1000-reverse.csv",'w')
    
    lines = file.readlines()
    lines= list(map(lambda x: x.rstrip("\n"), lines))
    rev = lines[::-1]
    lines2= file2.readlines()
    times= list(map(lambda x: x.split(',')[-1].rstrip("\n"), lines2))
    
    for line,time in zip(rev,times):
        file3.write("{},{},{},{}\n".format(line, int(line)%2, int(line)//2, time))
        
    file.close()
    file2.close()
    file3.close()

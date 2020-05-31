import codecs
import pandas as pd

def generarDataset(nombre="Enlaces/lista-temporal-10.csv", nombreSalida="Datasets/dataset10.csv", offset=6):
    vertices = list(map(lambda x: int(x.rstrip('\n')),codecs.open(nombre).readlines()))
    nfilas = len(vertices)
    dataset = []
    for i in range(nfilas-offset+1):
        dataset.append(vertices[i:i+offset])
        
    df = pd.DataFrame(dataset)
    df.to_csv(nombreSalida, index=False)


if __name__ == '__main__':
    generarDataset()
import pandas as pd
import numpy as np

def separarSecuestrosDesapariciones():
    df = pd.read_csv('victimas.csv')
    # Se separan los registros cuyo delito contiene SECUESTRO
    dfSecuestros = df[df['DELITO'].str.contains(pat="SECUESTRO")]
    # Se guardan los registros en el archivo ConteoVictimasSecuestros.csv
    dfSecuestros.to_csv('./ConteoVictimasSecuestros.csv',index=False)
    # Se separan los registros cuyo delito contiene DESAPARICION
    dfDesapariciones = df[df['DELITO'].str.contains(pat="DESAPARICION")]
    # Se guardan los registros en el archivo ConteoVictimasDesapariciones.csv
    dfDesapariciones.to_csv('./Desapariciones.csv',index=False)
    
def cleanF(targetFile='Diccionarios/Municipios.txt',sourceFile='Diccionarios/Municipios.txt.1'):
    with open(targetFile,'w') as m:
        with open(sourceFile,'r') as f:
            for i in f:
                m.write(i.replace(' ','_'))
                
def aCategoria(valor):    
    if not isinstance(valor,str): 
        if np.isnan(valor): return -1
        return valor
    else:
        v = valor.upper()
        if v in ['SI','ACTIVO','COLOMBIA','MASCULINO']:
            return 1
        elif v == 'SIN REGISTRO':
            return -1
        else: 
            return 0
        
def limpiarCategorias(archivo="ConteoVictimasSecuestros.csv",archivoSalida="DataSetSecuestros.csv"):
    dfSecuestros = pd.read_csv(archivo)
    dfSecuestros = dfSecuestros.drop(columns="DELITO")
    dfSecuestros = dfSecuestros.drop(columns="ANIO_ENTRADA")
    # Convertir categorias y limpiear datos
    for i in ['RUPTURA','CONEXO','ESTADO_NOTICIA','PAIS','IMPUTACION','CONDENA','ATIPICIDAD_INEXISTENCIA','ACUSACION','CAPTURA','SEXO_VICTIMA','PAIS_NACIMIENTO']:
        dfSecuestros[i] = dfSecuestros[i].apply(aCategoria)

    #dfSecuestros['ANIO_DENUNCIA'] = dfSecuestros['ANIO_DENUNCIA'].apply(lambda x: str(x))
    dfSecuestros['LEY'] = dfSecuestros['LEY'].apply(lambda x: str(x).upper())
    
    labels = ['DEPARTAMENTO','MUNICIPIO','ETAPA','PAIS','DEPARTAMENTO','MUNICIPIO','LEY','SECCIONAL']
    years  = ['ANIO_DENUNCIA','ANIO_HECHO'] #,'ANIO_ENTRADA'
    #dfSecuestros[['DEPARTAMENTO']] = dfSecuestros[['DEPARTAMENTO','']].apply(lambda x: str(x).replace(' ','_'))
    
    def fillGaps(arg):
        s = str(arg)
        if len(s) == 0 or s == 'nan':
            return 'SIN_REGISTRO'
        else:
            return s.replace(' ','_')
    
    def fillYears(year):
        s = str(year).rstrip('.0')
        l = len(s)
        if s=='nan':
            return 'SIN_REGISTRO'
        elif l<4:
            return s+('0'*(4-l))
        else:
            return s
    
    fGrupoEdades = open('Diccionarios/GrupoEdades.txt','r')
    grupoEdades = [i.replace('\n','') for i in fGrupoEdades]
    
    def grupoEdad(grupo):    
        if grupo == 'SIN REGISTRO':
            return 'SIN_REGISTRO'
        else:
            for categoria in grupoEdades:
                if categoria in grupo.replace(' ','_'):
                    return categoria
    
    for label in labels:
        dfSecuestros[label] = dfSecuestros[label].apply(fillGaps)
        
    for year in years:
        dfSecuestros[year] = dfSecuestros[year].apply(fillYears)
    
    dfSecuestros['GRUPO_EDAD_VICTIMA'] = dfSecuestros['GRUPO_EDAD_VICTIMA'].apply(grupoEdad)
    
    dfSecuestros.to_csv(archivoSalida, index=False)

def agruparVictimasAnios(aniosEntrada=5, aniosSalida=1, dataSet = "DataSetSecuestros.csv"):
    df = pd.read_csv(dataSet)
    
    dfConteoVictimas = df[['ANIO_HECHO','TOTAL_VICTIMAS']].groupby(['ANIO_HECHO']).sum()
    #dfConteoVictimas = dfConteoVictimas.loc[dfConteoVictimas['ANIO_HECHO']!= 'SIN_REGISTRO']
    
    group = list(map(lambda x: x[1]['TOTAL_VICTIMAS'] if x[0] != "SIN REGISTRO" else -1,dfConteoVictimas.iterrows()))
    
    to_df = {}
    
    for i in range(aniosEntrada):
        to_df['Input_'+str(i)] = []
    
    for i in range(aniosSalida):
        to_df['Output_'+str(i)] = []
        
    for i in range(0,len(group)-(aniosEntrada+aniosSalida)):
        inputs = group[i:i+aniosEntrada]
        for input_with_index in zip(range(aniosEntrada),inputs):
            index, the_input = input_with_index
            to_df['Input_'+str(index)].append(the_input)
        
        j=i+aniosEntrada
        outputs = group[j:j+aniosSalida]
        for output_with_index in zip(range(aniosSalida),outputs):
            index, the_output = output_with_index
            to_df['Output_'+str(index)].append(the_output)
    
    #for key in to_df.keys():
        #print(len(to_df[key]))
    
    the_dataframe = pd.DataFrame(to_df)
    the_dataframe.to_csv('{}{}x{}.csv'.format(dataSet.rstrip('.csv'),aniosEntrada,aniosSalida),index=False)


def secuestrosDepartamentos():
    df= pd.read_csv('ConteoVictimasSecuestros.csv')
    df = df[df.DEPARTAMENTO != 'POR ESTABLECER']
    df['DEPARTAMENTO'] = df['DEPARTAMENTO'].apply(lambda x: str(x).upper())
    df = df[['DEPARTAMENTO','TOTAL_VICTIMAS']].groupby('DEPARTAMENTO')['TOTAL_VICTIMAS'].sum()
    df.to_csv('SecuestrosPorDepartamentos.csv')
    
if __name__ == '__main__':
    #limpiarCategorias()
    #limpiarCategorias(archivo="ConteoVictimasDesapariciones.csv",archivoSalida="DataSetDesapariciones.csv")    
    #agruparVictimasAnios(aniosEntrada=5, aniosSalida=1, dataSet='DataSetDesapariciones.csv')
    #cleanF(targetFile='Diccionarios/Etapas.txt',sourceFile='Diccionarios/Etapas.txt.1')
    #cleanF(targetFile='Diccionarios/Leyes.txt',sourceFile='Diccionarios/Leyes.txt.1')
    #cleanF(targetFile='Diccionarios/Seccionales.txt',sourceFile='Diccionarios/Seccionales.txt.1')
    #secuestrosDepartamentos()
    pass
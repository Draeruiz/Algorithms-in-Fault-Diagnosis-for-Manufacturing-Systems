# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:43:40 2025

@author: Abner
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from scipy.fft import fft, fftfreq
import scipy.stats
import statistics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import cross_validate,GridSearchCV,RandomizedSearchCV
from scipy.stats import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Lectura del archivo de datos.

def procesar_carpeta(carpeta, numero):
    dataframes_fusionados = {}

    for archivo in os.listdir(carpeta):
        if archivo.endswith('.xlsx'): 
            ruta_completa = os.path.join(carpeta, archivo)
            df = pd.read_excel(ruta_completa)
            df['Nueva Columna'] = numero
            nombre_inicial = archivo.split('_')[0]
            clave = nombre_inicial
            if clave in dataframes_fusionados:
                dataframes_fusionados[clave] = pd.concat([dataframes_fusionados[clave], df], ignore_index=True)
            else:
                dataframes_fusionados[clave] = df
    
    return dataframes_fusionados

carpetas = ['Cilindro Normal','Fuga manguera','Sin tapa','Sin succion']

dataframes_fusionados = {}
for i, carpeta_actual in enumerate(carpetas):
    numero = i + 1
    dataframes_carpeta = procesar_carpeta(carpeta_actual, numero)
    for clave, df in dataframes_carpeta.items():
        if clave in dataframes_fusionados:
            dataframes_fusionados[clave] = pd.concat([dataframes_fusionados[clave], df], ignore_index=True)
        else:
            dataframes_fusionados[clave] = df

arrays = [np.array(lista_df) for lista_df in dataframes_fusionados.values()]

dt1,dt2,dt3,dt4,dt5,dt6,dt7,dt8,dt9,dt10=arrays

target_1= dt1[:, -1]

for i in range (10):
    exec(f'dt{i+1}=np.delete(dt{i+1}, -1, axis=1)')

#sens=[0,1,2,3,4,5]    
#for i in sens:
for i in range (10):
    exec(f'dt{i+1}_rms=[]')
    exec(f'dt{i+1}_prom=dt{i+1}.mean(axis=1)')
    exec(f'dt{i+1}_std=dt{i+1}.std(axis=1)')
    exec(f'dt{i+1}_skew= scipy.stats.skew(dt{i+1},axis=1)')
    exec(f'dt{i+1}_kurtosis= scipy.stats.kurtosis(dt{i+1},axis=1)')
    exec(f'dt{i+1}_fftrms =np.zeros((dt1.shape[0],1))')
    exec(f'dt{i+1}_fftprom =np.zeros((dt1.shape[0],1))')
    exec(f'dt{i+1}_fftkurtosis =np.zeros((dt1.shape[0],1))')
    exec(f'dt{i+1}_fftskew =np.zeros((dt1.shape[0],1))')
    exec(f'dt{i+1}_fftstd =np.zeros((dt1.shape[0],1))')
    for j in range (dt1.shape[0]):
        exec(f'y=dt{i+1}[j,:]')
        yf=fft(y)
        yf=abs(yf[1:int(len(yf)/2)])
        exec(f'dt{i+1}_fftprom[j]=yf.mean(axis=0)')
        exec(f'dt{i+1}_fftkurtosis[j]= scipy.stats.kurtosis(yf,axis=0)')
        exec(f'dt{i+1}_fftskew[j]= scipy.stats.skew(yf,axis=0)')
        exec(f'dt{i+1}_fftstd[j]= yf.std(axis=0)')
        exec(f'dt{i+1}_fftrms[j]=np.sqrt(np.mean(np.square(yf)))')
    exec(f'dt{i+1}=pd.DataFrame(dt{i+1})')
    exec(f'dt{i+1}_rms=(((dt{i+1}.iloc[:,1:]**2).sum(1))/(len(dt{i+1})/2))**(1/2)')
    exec(f'dt{i+1}_shape=dt{i+1}_rms/dt{i+1}_prom')
    exec(f'dt{i+1}_crest=((abs(dt{i+1})).max(axis=1))/(dt{i+1}_rms)')
    exec(f'dt{i+1}_impulse=((abs(dt{i+1})).max(axis=1))/(dt{i+1}_prom)')
    exec(f'dt{i+1}_fftprom=dt{i+1}_fftprom.reshape(dt1.shape[0])')
    exec(f'dt{i+1}_fftkurtosis=dt{i+1}_fftkurtosis.reshape(dt1.shape[0])')
    exec(f'dt{i+1}_fftskew=dt{i+1}_skew.reshape(dt1.shape[0])')   
    exec(f'dt{i+1}_fftstd=dt{i+1}_fftstd.reshape(dt1.shape[0])')
    exec(f'dt{i+1}_fftstd=dt{i+1}_fftstd.reshape(dt1.shape[0])')
    exec(f'dt{i+1}_fftrms=dt{i+1}_fftrms.reshape(dt1.shape[0])') 
    if i==0:
            df=pd.DataFrame({"DT1prom":dt1_prom,"DT1rms":dt1_rms, "DT1std":dt1_std,"DT1skew":dt1_skew,
                             "DT1kurtosis":dt1_kurtosis,"DT1shape":dt1_shape,"DT1crest":dt1_crest,
                             "DT1impulse":dt1_impulse,"DT1fftprom":dt1_fftprom,"DT1fftskew":dt1_fftskew,
                             "DT1fftkurtosis":dt1_fftkurtosis,"DT1fftstd":dt1_fftstd,"DT1fftrms":dt1_fftrms})
    else:
        exec(f'df["dt{i+1}prom"]=dt{i+1}_prom')
        exec(f'df["dt{i+1}rms"]=dt{i+1}_rms')
        exec(f'df["dt{i+1}std"]=dt{i+1}_std')
        exec(f'df["dt{i+1}skew"]=dt{i+1}_skew')
        exec(f'df["dt{i+1}kurtosis"]=dt{i+1}_kurtosis')
        exec(f'df["dt{i+1}shape"]=dt{i+1}_shape')
        exec(f'df["dt{i+1}crest"]=dt{i+1}_crest')
        exec(f'df["dt{i+1}impulse"]=dt{i+1}_impulse')
        exec(f'df["dt{i+1}fftprom"]=dt{i+1}_fftprom')
        exec(f'df["dt{i+1}fftskew"]=dt{i+1}_fftskew')
        exec(f'df["dt{i+1}fftkurtosis"]=dt{i+1}_fftkurtosis')
        exec(f'df["dt{i+1}fftstd"]=dt{i+1}_fftstd')
        exec(f'df["dt{i+1}fftrms"]=dt{i+1}_fftrms')

x=df

#df = df.drop(["DT1prom","DT1rms", "DT1std","DT1skew","DT1kurtosis","DT1shape","DT1crest",
#                "DT1impulse","DT1fftprom","DT1fftskew","DT1fftkurtosis","DT1fftstd","DT1fftrms"], axis=1)

x= df
sc=StandardScaler()
x=sc.fit_transform(x)
X_Modeled=x

'''
pca = PCA()
pca.fit(x)

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada')
plt.grid()
plt.show()
# Calcular la varianza acumulada
varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
# Establecer umbral
umbral = 0.95  # 95% de varianza acumulada
n_componentes = np.argmax(varianza_acumulada >= umbral) + 1
print(f"Número de componentes necesarios para alcanzar el {umbral * 100}% de varianza acumulada: {n_componentes}")
pca_n = PCA(n_componentes)
X_Modeled = pca_n.fit_transform(x)
Y=target_1
'''


'''
def backwardElimination(x, sl):
        numVars = len(x[0])
        for i in range(0, numVars):
            regressor_OLS = sm.OLS(Y, x).fit()
            maxVar = max(regressor_OLS.pvalues)#.astype(float)
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if (regressor_OLS.pvalues[j] == maxVar): #.astype(float)
                        x = np.delete(x, j, 1)
                        print (maxVar,j,sep='--->col=')
        #print(regressor_OLS.summary())
        return x

import statsmodels.api as sm
X1 = np.append(arr = np.ones((dt1.shape[0], 1)).astype(int), values = x, axis = 1)
SL = 0.05
X_opt = X1[:,:]
X_Modeled = backwardElimination(X_opt, SL)
'''

Numejem=100
for i in range (Numejem):     
        
    exec(f'x_train,x_test,y_train,y_test=train_test_split(X_Modeled,target_1,test_size=0.2,random_state=500)')
    
    '''
    #Cross Validation
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm

    param_grid = {
        'C': [0.01, 0.1, 1, 5, 10, 25, 50, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']  # Especificar el kernel aquí
    }
    
    
    grid_search = GridSearchCV(svm.SVC(probability=True), param_grid, cv=50)
    grid_search.fit(x_train, y_train)
    # Imprimir los mejores parámetros
    print("Mejores parámetros encontrados:")
    print(grid_search.best_params_)
      
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None,5,10,15,20,25,30,35,40,45,50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=50)
    grid_search.fit(x_train, y_train)
    # Imprimir los mejores parámetros
    print("Mejores parámetros encontrados:")
    print(grid_search.best_params_)
    
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    
    param_grid = {
        'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
                        26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=50)
    grid_search.fit(x_train, y_train)

    # Imprimir los mejores parámetros
    print("Mejores parámetros encontrados:")
    print(grid_search.best_params_)
    
    '''
    # SVM
    from sklearn import svm
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    modeloSVM = svm.SVC(C=10, kernel='rbf', gamma=0.1 ,probability=True)
    modeloSVM.fit(x_train, y_train)
    y_pred=modeloSVM.predict(x_test)
    f1=f1_score(y_test, y_pred,average='macro')
    p=precision_score(y_test, y_pred,average='macro')
    r=recall_score(y_test, y_pred, average='macro')
    acc=accuracy_score(y_test, y_pred)
    sns.set()
    f,ax=plt.subplots()
    C2=confusion_matrix(y_test,y_pred)
    print(C2)
    sns.heatmap(C2,annot=True,ax=ax,cbar=True)
    ax.margins(4,4)
    exec(f'ax.set_title("Matriz de confusión SVM {i+1}")')
    ax.set_xlabel('Valores reales')
    ax.set_ylabel('Valores Predichos')
    exec(f'Metricas{i+1}=np.array([["  ", "SVM"], ["P",p],["R",r],["F1",f1],["Acc",acc]])')
    
    # ARBOL DE DECISION
    from sklearn import tree
    criterio='gini'
    min_samp_le=2
    min_samp_sp=10
    max_dep=10
        
    modeloTree = tree.DecisionTreeClassifier(criterion=criterio, min_samples_leaf= min_samp_le,
                                             min_samples_split=min_samp_sp,max_depth=max_dep)
    modeloTree.fit(x_train,y_train)
    y_pred=modeloTree.predict(x_test)
    f1=f1_score(y_test, y_pred,average='macro')
    p=precision_score(y_test, y_pred,average='macro')
    r=recall_score(y_test, y_pred, average='macro')
    acc=accuracy_score(y_test, y_pred)
    sns.set()
    f,ax=plt.subplots()
    C2=confusion_matrix(y_test,y_pred)
    print(C2)
    sns.heatmap(C2,annot=True,ax=ax,cbar=True)
    ax.margins(4,4)
    exec(f'ax.set_title("Matriz de confusión Árbol de decisión {i+1}")')
    ax.set_xlabel('predecir')
    ax.set_ylabel('true')
    Results=np.array([["Decision Tree"],[p],[r],[f1],[acc]])
    exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
    
    #METODOS DE ENSAMBLE
    #BOOSTING
    from sklearn import ensemble
    modeloAB=ensemble.AdaBoostClassifier(base_estimator=modeloTree,n_estimators=40)
    modeloAB.fit(x_train,y_train)
    y_pred=modeloAB.predict(x_test)
    f1=f1_score(y_test, y_pred,average='macro')
    p=precision_score(y_test, y_pred,average='macro')
    r=recall_score(y_test, y_pred, average='macro')
    acc=accuracy_score(y_test, y_pred)
    sns.set()
    f,ax=plt.subplots()
    C2=confusion_matrix(y_test,y_pred)
    print(C2)
    sns.heatmap(C2,annot=True,ax=ax,cbar=True)
    ax.margins(4,4)
    exec(f'ax.set_title("Matriz de confusión BOOSTING {i+1}")')
    ax.set_xlabel('predecir')
    ax.set_ylabel('true')
    Results=np.array([['Boosting'],[p],[r],[f1],[acc]])
    exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
        
    #RANDOM FOREST
    modeloRF= ensemble.RandomForestClassifier(criterion=criterio, min_samples_leaf= min_samp_le,
                                             min_samples_split=min_samp_sp,max_depth=max_dep,n_estimators=40)
    modeloRF.fit(x_train,y_train)
    y_pred=modeloRF.predict(x_test)
    f1=f1_score(y_test, y_pred,average='macro')
    p=precision_score(y_test, y_pred,average='macro')
    r=recall_score(y_test, y_pred, average='macro')
    acc=accuracy_score(y_test, y_pred)
    print(f1)
    print(p)
    print(r)
    print(acc)
    sns.set()
    f,ax=plt.subplots()
    C2=confusion_matrix(y_test,y_pred)
    print(C2)
    sns.heatmap(C2,annot=True,ax=ax,cbar=True)
    ax.margins(4,4)
    exec(f'ax.set_title("Matriz de confusión Random Forest {i+1}")')
    ax.set_xlabel('predecir')
    ax.set_ylabel('true')
    Results=np.array([['Random Forest'],[p],[r],[f1],[acc]])
    exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
    
    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn= KNeighborsClassifier(n_neighbors=14, metric='manhattan')
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    f1=f1_score(y_test, y_pred,average='macro')
    p=precision_score(y_test, y_pred,average='macro')
    r=recall_score(y_test, y_pred, average='macro')
    acc=accuracy_score(y_test, y_pred)
    print(f1)
    print(p)
    print(r)
    print(acc)
    sns.set()
    f,ax=plt.subplots()
    C2=confusion_matrix(y_test,y_pred)
    print(C2)
    sns.heatmap(C2,annot=True,ax=ax,cbar=True)
    ax.margins(4,4)
    exec(f'ax.set_title("Matriz de confusión KNN {i+1}")')
    ax.set_xlabel('predecir')
    ax.set_ylabel('true')
    Results=np.array([['KNN'],[p],[r],[f1],[acc]])
    exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')

    #STACKING
    from sklearn.linear_model import LogisticRegression
    estimator = [('ModelTree', modeloTree),('ModelTreeAB', modeloAB),('random_forest', modeloRF),
                 ('svm_model', modeloSVM),('knn_model', knn)]
    modeloS= ensemble.StackingClassifier(estimators=estimator,final_estimator=LogisticRegression(),cv=None)
    modeloS.fit(x_train,y_train)
    y_pred=modeloS.predict(x_test)
    f1=f1_score(y_test, y_pred,average='macro')
    p=precision_score(y_test, y_pred,average='macro')
    r=recall_score(y_test, y_pred, average='macro')
    acc=accuracy_score(y_test, y_pred)
    print(f1)
    print(p)
    print(r)
    print(acc)
    sns.set()
    f,ax=plt.subplots()
    C2=confusion_matrix(y_test,y_pred)
    print(C2)
    sns.heatmap(C2,annot=True,ax=ax,cbar=True)
    ax.margins(4,4)
    exec(f'ax.set_title("Matriz de confusión STACKING {i+1}")')
    ax.set_xlabel('predecir')
    ax.set_ylabel('true')
    Results=np.array([['Stacking'],[p],[r],[f1],[acc]])
    exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
    
    #Inicializar Red Neuronal
    from keras.utils import to_categorical
    y_train=y_train-1
    y_train = to_categorical(y_train)
    RNA=Sequential() 
    #Capa de entrada
    RNA.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
    #Capa oculta
    #RNA.add(Dense(64,activation='relu'))
    RNA.add(Dense(32,activation='relu'))
    #Capa de salida
    RNA.add(Dense(units=y_train.shape[1], activation='softmax'))
    RNA.summary()
    RNA.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    RNA.fit(x_train,y_train,batch_size=8,epochs=50)
    y_pred=RNA.predict(x_test)
    
    y_pred = RNA.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred=y_pred+1

    f1=f1_score(y_test, y_pred,average='macro')
    p=precision_score(y_test, y_pred,average='macro')
    r=recall_score(y_test, y_pred, average='macro')
    acc=accuracy_score(y_test, y_pred)
    print(f1)
    print(p)
    print(r)
    print(acc)
    sns.set()
    f,ax=plt.subplots()
    C2=confusion_matrix(y_test,y_pred)
    print(C2)
    sns.heatmap(C2,annot=True,ax=ax,cbar=True)
    ax.margins(4,4)
    exec(f'ax.set_title("Matriz de confusión RNA {i+1}")')
    ax.set_xlabel('predecir')
    ax.set_ylabel('true')
    Results=np.array([['RNA'],[p],[r],[f1],[acc]])
    exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
    
    '''
    #Inicializar Red Neuronal
    RNA2=Sequential()
    #Capa de entrada
    RNA2.add(Dense(units=32, activation='relu', input_dim=x_train.shape[1]))
    #Capa oculta
    #RNA2.add(Dense(64,activation='relu'))
    RNA2.add(Dense(32,activation='relu'))
    #Capa de salida
    RNA2.add(Dense(units=y_train.shape[1], activation='softmax'))
    RNA2.summary()
    RNA2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    RNA2.fit(x_train,y_train,batch_size=8,epochs=100)
    y_pred=RNA2.predict(x_test)
    
    y_pred = RNA2.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred=y_pred+1
    
    f1=f1_score(y_test, y_pred,average='macro')
    p=precision_score(y_test, y_pred,average='macro')
    r=recall_score(y_test, y_pred, average='macro')
    acc=accuracy_score(y_test, y_pred)
    print(f1)
    print(p)
    print(r)
    print(acc)
    sns.set()
    f,ax=plt.subplots()
    C2=confusion_matrix(y_test,y_pred)
    print(C2)
    sns.heatmap(C2,annot=True,ax=ax,cbar=True)
    ax.margins(4,4)
    exec(f'ax.set_title("Matriz de confusión RNA {i+1}")')
    ax.set_xlabel('predecir')
    ax.set_ylabel('true')
    Results=np.array([['RNA2'],[p],[r],[f1],[acc]])
    exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
    '''
    print(i)
    
    
Metricasprom=Metricas1
for i in range (2,Numejem+1):
    for j in range (1,4):
        for k in range(1,2):
            exec(f'Metricasprom[j,k]=float(Metricasprom[j,k])+float(Metricas{i}[j,k])')
for j in range (1,4):
    for k in range(1,2):    
        Metricasprom[j,k]=float(Metricasprom[j,k])/Numejem
'''
Numejem=3
for i in range (Numejem):     
        
    exec(f'x_train,x_test,y_train,y_test=train_test_split(X_Modeled,target_1,test_size=0.8,random_state=500)')
    
    from sklearn.preprocessing import StandardScaler
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)
    
    #Convertir etiquetas a one-hot encoding
    from keras.utils import to_categorical
    y_train=y_train-1
    y_train = to_categorical(y_train) 
    from keras.models import Sequential
    from keras.layers import Dense
    import numpy as np
    ep=[10,20,50,100]
    bs=[2,8,32,128]
    optimizador = ['adam', 'sgd', 'RMSprop']
    for optim in optimizador:
        for epo in (ep):
            for bat in (bs):
                print(optim,epo,bat,i)
    # Crear modelo secuencial
                model = Sequential()
                model.add(Dense(units=32, activation='relu', input_dim=x_train.shape[1]))
                model.add(Dense(units=32, activation='relu'))
                model.add(Dense(units=3, activation='softmax'))  # Capa de salida con 3 unidades y activación softmax
# Compilar el modelo
                model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
# Entrenar el modelo
                model.fit(x_train, y_train, epochs=epo, batch_size=bat, validation_split=0.1, verbose=1)
# Ejemplo de predicción con nuevos datos
                y_pred = model.predict(x_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred=y_pred+1
                f1=f1_score(y_test, y_pred,average='weighted')
                p=precision_score(y_test, y_pred,average='weighted')
                r=recall_score(y_test, y_pred, average='weighted')
                acc=accuracy_score(y_test, y_pred)
# Mostrar resultados
                print(f1)
                print(p)
                print(r)
                print(acc)
                if epo==10 and bat==2 and optim=='adam':
                    exec(f'Metricas{i+1}=np.array([["  ", "RNA32_32_{optim}_{epo}_{bat}"], ["P",p],["R",r],["F1",f1],["Acc",acc]])')
                else:
                    exec(f'Results=np.array([["RNA32_32_{optim}_{epo}_{bat}"],[p],[r],[f1],[acc]])')
                    exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
                   
                model = Sequential()
                model.add(Dense(units=32, activation='relu', input_dim=x_train.shape[1]))
                model.add(Dense(units=32, activation='relu'))
                model.add(Dense(units=32, activation='relu'))
                model.add(Dense(units=3, activation='softmax'))  # Capa de salida con 3 unidades y activación softmax
# Compilar el modelo
                model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
# Entrenar el modelo
                model.fit(x_train, y_train, epochs=epo, batch_size=bat, validation_split=0.1, verbose=1)
# Ejemplo de predicción con nuevos datos
                y_pred = model.predict(x_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred=y_pred+1
                f1=f1_score(y_test, y_pred,average='weighted')
                p=precision_score(y_test, y_pred,average='weighted')
                r=recall_score(y_test, y_pred, average='weighted')
                acc=accuracy_score(y_test, y_pred)
# Mostrar resultados
                print(f1)
                print(p)
                print(r)
                print(acc)
                exec(f'Results=np.array([["RNA32_32_32_{optim}_{epo}_{bat}"],[p],[r],[f1],[acc]])')
                exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
                
                model = Sequential()
                model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
                model.add(Dense(units=64, activation='relu'))
                model.add(Dense(units=3, activation='softmax'))  # Capa de salida con 3 unidades y activación softmax
# Compilar el modelo
                model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
# Entrenar el modelo
                model.fit(x_train, y_train, epochs=epo, batch_size=bat, validation_split=0.1, verbose=1)
# Ejemplo de predicción con nuevos datos
                y_pred = model.predict(x_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred=y_pred+1
                f1=f1_score(y_test, y_pred,average='weighted')
                p=precision_score(y_test, y_pred,average='weighted')
                r=recall_score(y_test, y_pred, average='weighted')
                acc=accuracy_score(y_test, y_pred)
# Mostrar resultados
                print(f1)
                print(p)
                print(r)
                print(acc)
                exec(f'Results=np.array([["RNA64_64_{optim}_{epo}_{bat}"],[p],[r],[f1],[acc]])')
                exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
            
                model = Sequential()
                model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
                model.add(Dense(units=64, activation='relu'))
                model.add(Dense(units=64, activation='relu'))
                model.add(Dense(units=3, activation='softmax'))  # Capa de salida con 3 unidades y activación softmax
# Compilar el modelo
                model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
# Entrenar el modelo
                model.fit(x_train, y_train, epochs=epo, batch_size=bat, validation_split=0.1, verbose=1)
# Ejemplo de predicción con nuevos datos
                y_pred = model.predict(x_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred=y_pred+1
                f1=f1_score(y_test, y_pred,average='weighted')
                p=precision_score(y_test, y_pred,average='weighted')
                r=recall_score(y_test, y_pred, average='weighted')
                acc=accuracy_score(y_test, y_pred)
# Mostrar resultados
                print(f1)
                print(p)
                print(r)
                print(acc)
                exec(f'Results=np.array([["RNA64_64_64_{optim}_{epo}_{bat}"],[p],[r],[f1],[acc]])')
                exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
            
                model = Sequential()
                model.add(Dense(units=64, activation='relu', input_dim=x_train.shape[1]))
                model.add(Dense(units=32, activation='relu'))
                model.add(Dense(units=3, activation='softmax'))  # Capa de salida con 3 unidades y activación softmax
# Compilar el modelo
                model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
# Entrenar el modelo
                model.fit(x_train, y_train, epochs=epo, batch_size=bat, validation_split=0.1, verbose=1)
# Ejemplo de predicción con nuevos datos
                y_pred = model.predict(x_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred=y_pred+1
                f1=f1_score(y_test, y_pred,average='weighted')
                p=precision_score(y_test, y_pred,average='weighted')
                r=recall_score(y_test, y_pred, average='weighted')
                acc=accuracy_score(y_test, y_pred)
# Mostrar resultados
                print(f1)
                print(p)
                print(r)
                print(acc)
                exec(f'Results=np.array([["RNA64_32_{optim}_{epo}_{bat}"],[p],[r],[f1],[acc]])')
                exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
            
                model = Sequential()
                model.add(Dense(units=128, activation='relu', input_dim=x_train.shape[1]))
                model.add(Dense(units=64, activation='relu'))
                model.add(Dense(units=32, activation='relu'))
                model.add(Dense(units=3, activation='softmax'))  # Capa de salida con 3 unidades y activación softmax
# Compilar el modelo
                model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
# Entrenar el modelo
                model.fit(x_train, y_train, epochs=epo, batch_size=bat, validation_split=0.1, verbose=1)
# Ejemplo de predicción con nuevos datos
                y_pred = model.predict(x_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred=y_pred+1
                f1=f1_score(y_test, y_pred,average='weighted')
                p=precision_score(y_test, y_pred,average='weighted')
                r=recall_score(y_test, y_pred, average='weighted')
                acc=accuracy_score(y_test, y_pred)
# Mostrar resultados
                print(f1)
                print(p)
                print(r)
                print(acc)
                exec(f'Results=np.array([["RNA128_64_32_{optim}_{epo}_{bat}"],[p],[r],[f1],[acc]])')
                exec(f'Metricas{i+1}=np.append(Metricas{i+1}, Results, axis=1)')
                
for i in range (2)  :             
    exec(f'Metri=Metricas{i+1}[1:,1:]')
    Metri = Metri.astype(np.float64) 
    exec(f'maximos_por_fila{i+1} = np.max(Metri, axis=1)')
    exec(f'indices_maximos_por_fila{i+1} = np.argmax(Metri, axis=1)')
# Imprimir los resultados
    exec(f'print("Valores máximos por fila:", maximos_por_fila{i+1})')
    exec(f'print("Índices de los máximos por fila (columna donde se encuentra el máximo):", indices_maximos_por_fila{i+1})')
    exec(f'print(Metricas{i+1}[0,indices_maximos_por_fila{i+1}[0]+1])')
    exec(f'print(Metricas{i+1}[1,indices_maximos_por_fila{i+1}[0]+1])')
    exec(f'print(Metricas{i+1}[2,indices_maximos_por_fila{i+1}[0]+1])')
    exec(f'print(Metricas{i+1}[3,indices_maximos_por_fila{i+1}[0]+1])')
    exec(f'print(Metricas{i+1}[4,indices_maximos_por_fila{i+1}[0]+1])')
Metricasprom= Metricas1.copy()
for i in range (2,Numejem+1):
    for j in range (1,5):
        for k in range(1,289):
            exec(f'Metricasprom[j,k]=float(Metricasprom[j,k])+float(Metricas{i}[j,k])')
for j in range (1,5):
    for k in range(1,289):    
        Metricasprom[j,k]=float(Metricasprom[j,k])/Numejem
Metri=Metricasprom[1:,1:]  
Metri = Metri.astype(np.float64) 
maximos_por_fila = np.max(Metri, axis=1)
indices_maximos_por_fila = np.argmax(Metri, axis=1)
print("Valores máximos por fila:", maximos_por_fila)
print("Índices de los máximos por fila (columna donde se encuentra el máximo):", indices_maximos_por_fila)
print(Metricasprom[0,indices_maximos_por_fila[0]+1])
print(Metricasprom[1,indices_maximos_por_fila[0]+1])
print(Metricasprom[2,indices_maximos_por_fila[0]+1])
print(Metricasprom[3,indices_maximos_por_fila[0]+1])
print(Metricasprom[4,indices_maximos_por_fila[0]+1])
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine

"""
1. Cargamos el dataset
"""
brain = pd.read_csv("./Brain_GSE50161.csv", low_memory=False)
X = brain.iloc[:, 1:] #Analizamos todas las columnas excluyendo el ID de los ejemplares (primera columna)
datasetName= "Brain Cancer Dataset"
print(f"{datasetName}:\n", brain)

#Filtramos los tipos de cancer de cerebro y reemplazamos los Strings por numeros
X.loc[X["type"] == "ependymoma", "type"] = 1
X.loc[X["type"] == "glioblastoma", "type"] = 2
X.loc[X["type"] == "medulloblastoma", "type"] = 3
X.loc[X["type"] == "pilocytic_astrocytoma", "type"] = 4
X.loc[X["type"] == "normal", "type"] = 5
print(f"\nSlice de {datasetName}:\n", X)

"""
2. Calcular la matriz de correlación de Pearson
"""
corr_matrix = X.corr(method='pearson') #Dataframe con los valores de correlacion entre atributos
print("\nMatriz de correlacion de Pearson: \n", corr_matrix)
    #Note que los valores van de -1 a 1.
    #Si existe una relacion debil entre valores, tendremos un valor pequeño
    #Si existe una relacion fuerte, tendremos un valor alto
    #Cuando el valor es -1, tenemos la relacion mas fuerte con pendiente negativa
    #Cuando el valor es 1, tenemos la relacion mas fuerte con pendiente positiva
    #Si no se puede hacer pasar una linea recta a traves de todos los valores (en una grafica de dispersion),
    #entonces los valores de correlacion estaran mas cerca de 0.
    #Cuando el valor es 0, no existe una linea recta que represente adecuadamente la relacion entre variables.
    #Nuestra confianza al hacer inferencias usando la relacion lineal crece al aumentar el numero de datos,
    #y tambien crece al decrecer el valor p de nuestras inferencias.
    #Al tener mas datos, mas pequeño sera el valor P y mas confianza tendremos en nuestras inferencias.
    #El valor P nos indica la probabilidad de que exista una relacion lineal igual o mejor al agregar datos aleatoriamente.

"""
3. Mostrar el mapa de calor de correlaciones
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    # annot=True muestra los valores de correlacion en el mapa
    # cmap='coolwarm' es una manera de representar los colores en el mapa
    # fmt=".2f" formatea los valores a 2 decimales
plt.title(f"Matriz de Correlación - {datasetName}")
plt.show()
"""

"""
4. Encontrar pares de características altamente correlacionadas
"""
# Umbral de correlación (por ejemplo, 0.85)
threshold = 0.85

high_corr_features = set() #unordered, unchangeable, unindexed, duplicates not allowed

for i in range(len(corr_matrix.columns) - 1): # Por cada columna en la matriz de correlacion, excluyendo la columna 'target'
    for j in range(i + 1, len(corr_matrix.columns) - 1):
        if abs(corr_matrix.iloc[i, j]) > threshold: # Checa si cada elemento es mayor al umbral
            colname = corr_matrix.columns[j] # Si es mayor al umbral, obtiene el nombre de la columna (atributo)
            high_corr_features.add(colname) # y agrega el nombre al set de atributos de alta correlacion

"""
5. Eliminar características altamente correlacionadas
"""
X_filtered = X.drop(columns=high_corr_features)

"""
6. Guardar el nuevo dataset en un archivo CSV
"""
csv_filename = "brain_filtrado_Pearson.csv"
X_filtered.to_csv(csv_filename, index=False)

"""
7. Mostrar características eliminadas y dimensiones
"""
print(f"\nCaracterísticas eliminadas por alta correlación: {high_corr_features}")
print(f"Dimensiones originales: {X.shape}")
print(f"Dimensiones después del filtrado: {X_filtered.shape}")
print(f"\nNuevo dataset guardado en '{csv_filename}'")

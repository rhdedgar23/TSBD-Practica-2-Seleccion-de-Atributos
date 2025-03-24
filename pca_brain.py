import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

"""
# 1. Cargar el dataset desde un archivo CSV
try:
    df = pd.read_csv('train.csv')  
except FileNotFoundError:
    print("Error: El archivo CSV no se encontró.")
    exit()
"""

"""
1. Cargar el dataset

brain = pd.read_csv("./Brain_GSE50161.csv", low_memory=False)
brain_slice = brain.iloc[:, 1:21] #Analizamos la primeras 20 columnas excluyendo el ID de los ejemplares
#Filtramos los tipos de cancer de cerebro y reemplazamos los Strings por numeros
brain_slice.loc[brain_slice["type"] == "ependymoma", "type"] = 1
brain_slice.loc[brain_slice["type"] == "glioblastoma", "type"] = 2
brain_slice.loc[brain_slice["type"] == "medulloblastoma", "type"] = 3
brain_slice.loc[brain_slice["type"] == "pilocytic_astrocytoma", "type"] = 4
brain_slice.loc[brain_slice["type"] == "normal", "type"] = 5
X = brain_slice.iloc[:, 1:21]
y = brain_slice.type
datasetName= "Brain Cancer Dataset"
print(f"{datasetName} Slice:\n", brain_slice)
print("\nCaracteristicas:\n", X)
print("\nClases:\n", y)
"""

"""
1. Cargar el dataset
"""
brain = pd.read_csv("./Brain_GSE50161.csv", low_memory=False)
#Filtramos los tipos de cancer de cerebro y reemplazamos los Strings por numeros
brain.loc[brain["type"] == "ependymoma", "type"] = 1
brain.loc[brain["type"] == "glioblastoma", "type"] = 2
brain.loc[brain["type"] == "medulloblastoma", "type"] = 3
brain.loc[brain["type"] == "pilocytic_astrocytoma", "type"] = 4
brain.loc[brain["type"] == "normal", "type"] = 5
X = brain.iloc[:, 2:]
y = brain.type
datasetName= "Brain Cancer Dataset"
print(f"{datasetName}:\n", brain)
print("\nCaracteristicas:\n", X)
print("\nClases:\n", y)

"""
2. Normalizar los datos
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print("\nCracteristicas normalizadas: ", X_scaled)

"""
3. Aplicar PCA sin reducir dimensiones (para analizar la varianza acumulada)
"""
pca_full = PCA() #se crea una instancia del modelo PCA
pca_full.fit(X_scaled)
# Usamos fit() en vez de fit_transform como en LDA porque en el modelo solo necesita saber la direccion de maxima varianza
# y no necesariamente necesitamos proyectar los datos (transform) a los componentes principales
# En LDA usualmente necesitamos esa proyeccion (transform) de manera inmediata.

"""
4. Calcular la varianza explicada acumulada
"""
# En PCA no se calcula la varianza explicada de manera manual
# este calculo ya es forma parte del modelo PCA(),
# ya que este modelo se enfoca en encontrar la maxima varianza para determinada variable en la combinacion  lineal.
# a diferencia de LDA que se enfoca en encontrar la maxima separabilidad de clases.
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
print("\nSuma acumulativa por cada componente: ", explained_variance)

"""
5. Graficar la varianza acumulada
"""
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='-')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Explicada por PCA')
plt.axhline(y=0.90, color='r', linestyle='--', label="90% de Varianza")
plt.axhline(y=0.95, color='g', linestyle='--', label="95% de Varianza")
plt.legend()
plt.grid()
plt.show()

"""
6. Seleccionar el número óptimo de componentes para retener al menos el 90% de la varianza
"""
n_components_opt = np.argmax(explained_variance >= 0.90) + 1
print(f"\nNúmero óptimo de componentes para retener 90% de la varianza: {n_components_opt}")

"""
7. Aplicar PCA con el número óptimo de componentes
"""
pca = PCA(n_components=n_components_opt)
X_pca = pca.fit_transform(X_scaled)

"""
Los datos nuevos quedaron en X_pca, es decir ya transformados en un dataset de 178 x 8
para el ejemplo del wine dataset. Pero en este paso 8 lo que se hace es darle el formato
de DataFrame poniéndole los títulos a las nuevas columnas.
"""
"""
8. Crear un nuevo dataset con la dimensión reducida
"""
df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components_opt)])
df_pca['Clase'] = y  # Agregar la columna de clases

"""
9. Mostrar el nuevo dataset
"""
print(f"\nDimensión original de X: {X_scaled.shape}")
print(f"\nDimensión reducida de X después de PCA: {X_pca.shape}\n")
print(df_pca)

"""
10. Guardar el dataset reducido en un archivo CSV
"""
df_pca.to_csv('brain_pca_reducido.csv', index=False)
print("\nDataset reducido guardado como 'brain_pca_reducido.csv'")

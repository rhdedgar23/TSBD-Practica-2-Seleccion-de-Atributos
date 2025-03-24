import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

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
y = brain.type.astype(int)
datasetName= "Brain Cancer Dataset"
print(f"{datasetName}:\n", brain)
print("\nCaracteristicas:\n", X)
print("\nClases:\n", y)

"""
2. Normalizar los datos
"""
scaler = StandardScaler()#StandardScaler() normaliza los datos. Media=0. Varianza=1.
X_scaled = scaler.fit_transform(X)#fit to data, then transform it
#print("\nCracteristicas normalizadas: ", X_scaled)

"""
3. Aplicar LDA sin reducción inicial para analizar la varianza explicada
"""
lda_full = LDA()#se crea una instancia del modelo LDA sin numero de componentes, para analizar la varianza explicada
X_lda_full = lda_full.fit_transform(X_scaled, y) #para analizar la varianza explicada por cada discriminante.
#como LDA es aprendizaje supervisado, necesitas caracteristica X y clases Y
#print("\nLDA aplicado a caracteristicas para analisis de varianza: ", X_lda_full)

"""
4. Calcular la varianza explicada por cada componente
"""
explained_variance = np.var(X_lda_full, axis=0) / np.sum(np.var(X_lda_full, axis=0))
# np.var(X_lda_full, axis=0) Calcula la varianza a lo largo de cada discriminante en X_lda_full
#Esto nos da una arreglo de varianzas, un arreglo por discriminante.
#np.sum(np.var(X_lda_full, axis=0)) suma las varianzas para obtener la varianza total.
# Cada varianza individual es dividida por la varianza total,
#lo cual nos da una "proporcion de varianza explicada" por cada discriminante lineal.
print("\nVarianza epxlicada por cada componente: ", explained_variance)
explained_variance_cumsum = np.cumsum(explained_variance)#Calcula la suma acumulativa del arreglo mencionado arriba.
print("\nSuma acumulativa por cada componente: ", explained_variance_cumsum)

"""
# 5. Graficar la varianza acumulada
"""
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_cumsum) + 1), explained_variance_cumsum, marker='o', linestyle='-')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Explicada por LDA')
plt.axhline(y=0.90, color='r', linestyle='--', label="90% de Varianza")
plt.axhline(y=0.95, color='g', linestyle='--', label="95% de Varianza")
plt.legend()
plt.grid()
plt.show()

"""
6. Determinar el número óptimo de componentes para retener al menos 90% de la varianza
"""
n_components_opt = np.argmax(explained_variance_cumsum >= 0.90) + 1 # +1 porque argmax() regresa el indice
# y el tamaño de explained_variance_cumsum es el numero de componentes
print(f"\nNúmero óptimo de componentes para retener 90% de la varianza: {n_components_opt}")

"""
7. Aplicar LDA con el número óptimo de componentes
"""
lda = LDA(n_components=n_components_opt)
X_lda = lda.fit_transform(X_scaled, y)

"""
8. Crear un nuevo dataset con la dimensión reducida
"""
df_lda = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(n_components_opt)]) #nombramos las columnas de los componentes
df_lda['Clase'] = y  # y gregamos la columna de clases

"""
9. Mostrar el nuevo dataset
"""
print(f"\nDimensión original de X: {X_scaled.shape}")
print(f"\nDimensión reducida de X después de LDA: {X_lda.shape}\n")
print(df_lda)

"""
10. Guardar el dataset reducido en un archivo CSV
"""
df_lda.to_csv('brain_lda_reducido.csv', index=False)
print("\nDataset reducido guardado como 'brain_lda_reducido.csv'")

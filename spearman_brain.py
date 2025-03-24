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
2. Calcular la matriz de correlación de Spearman
"""
corr_matrix = X.corr(method='spearman') #Dataframe con los valores de correlacion entre atributos
print("\nMatriz de correlacion de Spearman: \n", corr_matrix)

"""
3. Mostrar el mapa de calor de correlaciones
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Matriz de Correlación - Wine Dataset (Spearman)")
plt.show()
"""

"""
4. Encontrar pares de características altamente correlacionadas
"""
# Umbral de correlación (por ejemplo, 0.85)
threshold = 0.85

# Encontrar pares de características altamente correlacionadas
high_corr_features = set()

for i in range(len(corr_matrix.columns) - 1):  # Excluir la columna 'target'
    for j in range(i + 1, len(corr_matrix.columns) - 1):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[j]
            high_corr_features.add(colname)

"""
5. Eliminar características altamente correlacionadas
"""
X_filtered = X.drop(columns=high_corr_features)

"""
6. Guardar el nuevo dataset en un archivo CSV
"""
csv_filename = "brain_filtrado_Spearman.csv"
X_filtered.to_csv(csv_filename, index=False)

"""
7. Mostrar características eliminadas y dimensiones
"""
print(f"Características eliminadas por alta correlación (Spearman): {high_corr_features}")
print(f"Dimensiones originales: {X.shape}")
print(f"Dimensiones después del filtrado: {X_filtered.shape}")
print(f"Nuevo dataset guardado en '{csv_filename}'")

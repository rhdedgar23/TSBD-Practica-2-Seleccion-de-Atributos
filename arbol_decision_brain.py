import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

"""
1. Cargamos el dataset original

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

"""
1. Cargamos el dataset despues de la reduccion con PCA

brain = pd.read_csv("./brain_pca_reducido.csv", low_memory=False)
X = brain.iloc[:, :-1]
y = brain["Clase"]
print("Brain Dataset: \n", brain)
print("\nCaracteristicas:\n", X)
print("\nClases:\n", y)
"""

"""
1. Cargamos el dataset despues de la reduccion con LDA

brain = pd.read_csv("./brain_lda_reducido.csv", low_memory=False)
X = brain.iloc[:, :-1]
y = brain["Clase"]
print("Brain Dataset: \n", brain)
print("\nCaracteristicas:\n", X)
print("\nClases:\n", y)
"""

"""
1. Cargamos el dataset despues del filtrado con Pearson

brain = pd.read_csv("./brain_filtrado_Pearson.csv", low_memory=False)
X = brain.iloc[:, 1:]
y = brain["type"]
print("Wine Dataset: \n", brain)
print("\nCaracteristicas:\n", X)
print("\nClases:\n", y)
"""


"""
1. Cargamos el dataset despues del filtrado con Spearman
"""

brain = pd.read_csv("./brain_filtrado_Spearman.csv", low_memory=False)
X = brain.iloc[:, 1:]
y = brain["type"]
print("Wine Dataset: \n", brain)
print("\nCaracteristicas:\n", X)
print("\nClases:\n", y)


"""
2. Dividimos los datos en entrenamiento (80%) y prueba (20%)
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

"""
3. Crear y entrenar el modelo de árbol de decisión
"""
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

"""
4. Hacer predicciones en el conjunto de prueba
"""
y_pred = tree_model.predict(X_test)

"""
5. Calcular la precisión
"""
accuracy = accuracy_score(y_test, y_pred)

"""
6. Mostrar resultado
"""
print(f"Precisión del Árbol de Decisión: {accuracy:.4f}")

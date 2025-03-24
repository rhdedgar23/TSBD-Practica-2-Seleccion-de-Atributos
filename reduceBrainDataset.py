import pandas as pd

brain = pd.read_csv("./Brain_GSE50161.csv", low_memory=False)
brainReduced = brain.iloc[:, :5675]

X = brain.iloc[:, 2:]
y = brain.type

datasetName= "Brain Cancer Dataset"
print(f"{datasetName}:\n", brainReduced)
print("\nCaracteristicas:\n", X)
print("\nClases:\n", y)

brainReduced.to_csv('Brain_GSE50161.csv', index=False)
print("\nDataset reducido guardado como 'Brain_GSE50161.csv'")
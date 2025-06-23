import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Charger les données
fichier = "Data_PE_2025-CSI3_CIR3.xlsx"
df = pd.read_excel(fichier)
df = df.dropna()  # Supprime les lignes avec des valeurs manquantes
# Sélectionner les colonnes pertinentes
data = df[['Données 1','Données 2','Données 3','Données 4','Données 5']]
for i in range (len(data.columns)):
    for j in range(len(data.columns)):

"""
# Régression linéaire
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)

# Coefficients
b1 = reg.coef_[0]
b0 = reg.intercept_
r2 = r2_score(y, y_pred)

# Affichage
plt.scatter(X, y, color='blue', label='Données')
plt.plot(X, y_pred, color='red', label='Régression')
plt.xlabel("Données 1")
plt.ylabel("Données 5")
plt.title(f"Régression linéaire : y = {b0:.2f} + {b1:.2f}x\nR² = {r2:.2f}")
plt.legend()
plt.grid(True)
plt.show()
"""
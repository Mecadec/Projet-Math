import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Charger les données
fichier = "Data_PE_2025-CSI3_CIR3.xlsx"
df = pd.read_excel(fichier)
df = df.dropna()  # Supprime les lignes avec des valeurs manquantes

# Ignorer la première colonne
df_data = df.iloc[:, 1:]

# Récupérer les données dans une seule liste
data_list = df_data.values.flatten().tolist()
# Formater chaque valeur sous la forme (entier, décimal)
formatted_list = [(int(str(v).split('.')[0]), int(str(v).split('.')[1]) if '.' in str(v) else 0) for v in data_list]

# Créer les listes X et y
X = [val[0] for val in formatted_list]
y = [val[1] for val in formatted_list]

# Adapter X et y pour la régression linéaire
X = np.array(X).reshape(-1, 1)
y = np.array(y)

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
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Régression linéaire : y = {b0:.2f} + {b1:.2f}x\nR² = {r2:.2f}")
plt.legend()
plt.grid(True)
plt.show()

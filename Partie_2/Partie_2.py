import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

# Charger les données
fichier = "Data_PE_2025-CSI3_CIR3.xlsx"
df = pd.read_excel(fichier)
df = df.dropna()  # Supprime les lignes avec des valeurs manquantes

# Ignorer la première colonne (titres des lignes)
data = df.iloc[:, 1:]

# Créer le dossier 'graph' s'il n'existe pas
graph_dir = os.path.join(os.path.dirname(__file__), "graph")
os.makedirs(graph_dir, exist_ok=True)

# Régression linéaire multiple par colonne
for target_col in data.columns:
    X = data.drop(columns=[target_col])
    y = data[target_col]
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"R² pour {target_col}: {r2:.4f}")
    # Visualisation des résultats
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5, label="Nuage de points (réel vs prédit)")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="Droite idéale y = ŷ")
    plt.title(f"Régression linéaire pour {target_col}")
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.text(0.05, 0.95, f"R² = {r2:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.legend()
    # Sauvegarde du graphique
    plt.savefig(os.path.join(graph_dir, f"regression_multiple_{target_col}.png"), bbox_inches='tight')
    plt.close()
    # Pour chaque variable explicative, afficher la droite de régression simple
    for feature_col in X.columns:
        x_feat = X[feature_col].values.reshape(-1, 1)
        reg_simple = LinearRegression()
        reg_simple.fit(x_feat, y)
        y_pred_simple = reg_simple.predict(x_feat)
        b0 = reg_simple.intercept_
        b1 = reg_simple.coef_[0]
        r2_simple = r2_score(y, y_pred_simple)
        plt.figure(figsize=(8, 5))
        plt.scatter(x_feat, y, alpha=0.5, label="Données")
        plt.plot(x_feat, y_pred_simple, color='red', label=f"Régression: ŷ = {b0:.2f} + {b1:.2f}x")
        plt.xlabel(feature_col)
        plt.ylabel(target_col)
        plt.title(f"Droite de régression de {target_col} selon {feature_col}")
        plt.text(0.05, 0.95, f"R² = {r2_simple:.3f}", transform=plt.gca().transAxes, fontsize=11, verticalalignment='top')
        plt.legend()
        # Sauvegarde du graphique
        plt.savefig(os.path.join(graph_dir, f"regression_{target_col}_vs_{feature_col}.png"), bbox_inches='tight')
        plt.close()
#        plt.show()

# Exemple avec les points M1(1,1), M2(1,2), M3(1,5), M4(3,4), M5(4,3), M6(6,2), M7(0,4)
points = np.array([
    [1, 1],
    [1, 2],
    [1, 5],
    [3, 4],
    [4, 3],
    [6, 2],
    [0, 4]
])
x = points[:, 0]
y = points[:, 1]

# Calcul des moyennes
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calcul de la pente b1 et de l'ordonnée à l'origine b0 (méthode des moindres carrés)
numerateur = np.sum((x - x_mean) * (y - y_mean))
denominateur = np.sum((x - x_mean) ** 2)
b1 = numerateur / denominateur
b0 = y_mean - b1 * x_mean

# Calcul des valeurs prédites
y_pred = b0 + b1 * x

# Calcul du coefficient de détermination R²
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
r2 = 1 - ss_res / ss_tot

print("Exemple sur les points donnés :")
print(f"Formule de la pente : b1 = Σ(xi-x̄)(yi-ȳ) / Σ(xi-x̄)² = {b1:.4f}")
print(f"Formule de l'ordonnée à l'origine : b0 = ȳ - b1*x̄ = {b0:.4f}")
print(f"Équation de la droite de régression : ŷ = {b0:.2f} + {b1:.2f}x")
print(f"Formule du R² : 1 - Σ(yi-ŷi)² / Σ(yi-ȳ)² = {r2:.4f}")

# Tracé et sauvegarde du graphique
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Points')
plt.plot(x, y_pred, color='red', label=f"Régression: ŷ = {b0:.2f} + {b1:.2f}x")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Droite de régression sur l'exemple donné")
plt.text(0.05, 0.95, f"R² = {r2:.3f}", transform=plt.gca().transAxes, fontsize=11, verticalalignment='top')
plt.legend()
plt.savefig(os.path.join(graph_dir, "exemple_points_regression.png"), bbox_inches='tight')
plt.close()

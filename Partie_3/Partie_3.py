import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Charger les données
a = pd.read_excel("Data_PE_2025-CSI3_CIR3.xlsx")
df = a.dropna(how='all')

# Nettoyer noms colonnes
df.columns = [col.strip() for col in df.columns]

# Nombre de colonnes de données
num_data = 9

# Générer toutes les combinaisons possibles (sans doublon et sans inverses)
combinaisons = list(itertools.combinations(range(1, num_data + 1), 2))

# Liste pour stocker les résultats
resultats = []

# Boucle sur toutes les combinaisons
for i, j in combinaisons:
    x_col = f"Données {i}"
    y_col = f"Données {j}"
    
    x = df[x_col].values
    y = df[y_col].values
    
    n = len(x)
    
    # Calcul b1 et b0
    b1 = np.sum( (x - np.mean(x)) * (y - np.mean(y)) ) / np.sum( (x - np.mean(x))**2 )
    b0 = np.mean(y) - b1 * np.mean(x)
    
    # Prédictions
    y_pred = b0 + b1 * x
    
    # Résidus et SCE
    residuals = y - y_pred
    SCE = np.sum(residuals**2)
    
    # MSE
    MSE = SCE / (n - 2)
    
    # Écart-type
    std_error = np.sqrt(MSE)
    
    # R²
    SS_tot = np.sum( (y - np.mean(y))**2 )
    R2 = 1 - SCE / SS_tot
    
    # Conclusion simple
    if R2 > 0.7:
        conclusion = "Bon ajustement"
    else:
        conclusion = "Mauvais ajustement"
    
    # Affichage console
    print(f"Régression de {y_col} en fonction de {x_col} :")
    print(f"b₀ : {b0:.4f}, b₁ : {b1:.4f}, SCE : {SCE:.4f}, MSE : {MSE:.4f}, Écart-type : {std_error:.4f}, R² : {R2:.4f}")
    print(f"Interprétation : {conclusion}\n")
    
    # Sauvegarder les résultats
    resultats.append({
        'x': x_col,
        'y': y_col,
        'b0': b0,
        'b1': b1,
        'SCE': SCE,
        'MSE': MSE,
        'Écart-type': std_error,
        'R2': R2,
        'Interprétation': conclusion
    })
    
    # Tracer le graphique
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x, y=y, color='blue', label='Données observées')
    plt.plot(x, y_pred, color='red', label='Droite de régression')
    plt.title(f"Régression de {y_col} en fonction de {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.grid(True)
    
    # Enregistrer le graphique
    plt.savefig(f"Graph/Regression_{i}_vers_{j}.png")
    plt.close()

# Convertir les résultats en DataFrame
df_resultats = pd.DataFrame(resultats)

# Sauvegarder dans un fichier CSV
df_resultats.to_csv("resultats_regression.csv", index=False)

print("Analyse terminée. Résultats sauvegardés dans 'resultats_regression.csv'.")

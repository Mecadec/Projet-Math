import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import pandas as pd

# Charger les données depuis le fichier Excel
fichier = "Partie_4/Data_PE_2025-CSI3_CIR3.xlsx"
df = pd.read_excel(fichier).dropna()  # Suppression des lignes avec NaN
df = df.iloc[:, 1:]  # Ignorer la première colonne (noms ou indices)

# Stocker les résultats pour chaque variable cible
results_summary = {}

# Effectuer la régression linéaire pour chaque variable cible, l'une après l'autre
for target_col in df.columns:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    n = len(y)
    
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    
    y_pred = model.predict(X_const)
    residuals = y - y_pred
    SCE = np.sum(residuals**2)
    MSE = SCE / (n - X_const.shape[1])
    s = np.sqrt(MSE)

    b = model.params
    SE = model.bse
    t_stats = model.tvalues
    p_values = model.pvalues
    
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - X_const.shape[1])
    conf_int = model.conf_int(alpha=alpha)
    
    results_summary[target_col] = {
        'coefficients': b,
        'std_errors': SE,
        't_stats': t_stats,
        'p_values': p_values,
        'conf_intervals': conf_int,
        's': s
    }

print(results_summary)

"""
# Exemple avec les points M1(1,1), M2(1,2), M3(1,5), M4(3,4), M5(4,3), M6(6,2), M7(0,4)
# Points donnés
x = np.array([1, 1, 1, 3, 4, 6, 0])
y = np.array([1, 2, 5, 4, 3, 2, 4])
n = len(x)

# Ajustement du modèle de régression linéaire
X = sm.add_constant(x)  # Ajout d'une constante pour b0
model = sm.OLS(y, X).fit()

# Coefficients
b0, b1 = model.params

# Prédictions et résidus
y_pred = model.predict(X)
residuals = y - y_pred

# Somme des carrés des erreurs
SCE = np.sum(residuals**2)

# MSE et écart-type des erreurs
MSE = SCE / (n - 2)
s = np.sqrt(MSE)

# Moyenne de x
x_mean = np.mean(x)

# Erreurs standards
SE_b1 = s / np.sqrt(np.sum((x - x_mean)**2))
SE_b0 = s * np.sqrt(1/n + x_mean**2 / np.sum((x - x_mean)**2))

# Statistiques t
t_b1 = b1 / SE_b1
t_b0 = b0 / SE_b0

# p-valeurs
p_b1 = 2 * (1 - stats.t.cdf(np.abs(t_b1), df=n-2))
p_b0 = 2 * (1 - stats.t.cdf(np.abs(t_b0), df=n-2))

# Intervalles de confiance à 95%
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df=n-2)
IC_b1 = (b1 - t_crit * SE_b1, b1 + t_crit * SE_b1)
IC_b0 = (b0 - t_crit * SE_b0, b0 + t_crit * SE_b0)

print((b0, b1), (SE_b0, SE_b1), (t_b0, t_b1), (p_b0, p_b1), (IC_b0, IC_b1), s)
"""


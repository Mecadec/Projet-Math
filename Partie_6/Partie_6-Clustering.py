# Importation des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

# 1. Charger les données
df = pd.read_excel("Data_PE_2025-CSI3_CIR3.xlsx")

# Afficher les premières lignes pour vérifier
print(df.head())

# 2. Prétraitement : standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. ACP (PCA) - réduction en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Affichage PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', s=50, alpha=0.7)
plt.title("Projection ACP (PCA) - 2 composantes principales")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.show()

# 4. t-SNE - visualisation en 2D
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Affichage t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='green', s=50, alpha=0.7)
plt.title("Projection t-SNE (2D)")
plt.grid(True)
plt.show()

# 5. Heatmap des distances
# Calcul des distances euclidiennes
distance_matrix = squareform(pdist(X_scaled, metric='euclidean'))

# Affichage heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, cmap="viridis")
plt.title("Heatmap des distances entre observations")
plt.show()
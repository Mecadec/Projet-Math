# --- Imports ---------------------------------------------------------
import numpy as np
import math
from pprint import pprint

# Try to import SciPy for dendrogram; if unavailable we'll skip plotting
have_scipy = True
try:
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, dendrogram
    import matplotlib.pyplot as plt
except ImportError:
    have_scipy = False
    import matplotlib.pyplot as plt  # still for potential manual plots

# --- Data ------------------------------------------------------------
points = {
    "M1": (1, 1),
    "M2": (1, 2),
    "M3": (1, 5),
    "M4": (3, 4),
    "M5": (4, 3),
    "M6": (6, 2),
    "M7": (0, 4),
}

labels = list(points.keys())
coords = np.array(list(points.values()))
n = len(points)

# --- Q1 : Distance functions ----------------------------------------
def dist(p, q):
    """Euclidean distance"""
    return math.dist(p, q)

def dist1(p, q):
    """Manhattan (L1) distance"""
    return abs(p[0]-q[0]) + abs(p[1]-q[1])

def dist_inf(p, q):
    """Chebyshev (L∞) distance"""
    return max(abs(p[0]-q[0]), abs(p[1]-q[1]))

# Ward distance explanation will be in narrative (not code)

# --- Q2 : dist_min ---------------------------------------------------
def dist_min(points_arr):
    """Return indices of the pair at minimal Euclidean distance"""
    min_d = math.inf
    best_pair = (-1, -1)
    for i in range(len(points_arr)):
        for j in range(i+1, len(points_arr)):
            d = dist(points_arr[i], points_arr[j])
            if d < min_d:
                min_d = d
                best_pair = (i, j)
    return best_pair, min_d

pair_indices, min_distance = dist_min(coords)
print("Q2 - Closest pair:", labels[pair_indices[0]], labels[pair_indices[1]], "; distance = ", round(min_distance, 3))

# --- Q3 : squared Euclidean distance matrix --------------------------
dist_matrix = squareform(pdist(coords, metric='euclidean')) if have_scipy else np.zeros((n, n))
print("\nQ3 - Squared Euclidean distance matrix:")
print(np.round(dist_matrix, 2))

# Identify first class Γ1
print("\nFirst merge (Γ1):", labels[pair_indices[0]], "+", labels[pair_indices[1]])

# --- Q4-5 : single‑linkage manual clustering -------------------------
# We'll perform agglomerative clustering with single linkage to replicate "méthode du plus proche voisin"
def single_linkage_step(clusters):
    """Perform one merge step using single linkage and return updated clusters and distance"""
    best_pair = None
    best_dist = math.inf
    # clusters is list of lists of indices
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            # distance between two clusters = min distance between their members
            for a in clusters[i]:
                for b in clusters[j]:
                    d = dist(coords[a], coords[b])
                    if d < best_dist:
                        best_dist = d
                        best_pair = (i, j)
    # merge clusters
    new_clusters = []
    for k, cl in enumerate(clusters):
        if k not in best_pair:
            new_clusters.append(cl)
    merged = clusters[best_pair[0]] + clusters[best_pair[1]]
    new_clusters.append(merged)
    return new_clusters, best_dist, best_pair

clusters = [[i] for i in range(n)]
merges = []
while len(clusters) > 1:
    clusters, merge_dist, idx_pair = single_linkage_step(clusters)
    merges.append((idx_pair, merge_dist, [list(cl) for cl in clusters]))

print("\nSequential merges (single linkage):")
for step, (pair, d, cls) in enumerate(merges, 1):
    print(f"  Step {step}: merge clusters {pair} at distance {d:.3f} -> {cls}")

# --- Q6-7 : automated dendrogram (if SciPy available) ---------------
if have_scipy:
    Z = linkage(coords, method='single', metric='euclidean')
    plt.figure(figsize=(6, 4))
    dendrogram(Z, labels=labels, orientation='top')
    plt.title("Hierarchical Clustering Dendrogram (Single Linkage)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()
else:
    print("\nSciPy not available; dendrogram skipped.")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from pathlib import Path
import openpyxl

from sklearn.decomposition import PCA

# --- Chargement -------------------------------------------------
# Chemin robuste : dossier du script → parent → data → fichier
root = Path(__file__).resolve().parent
file = root.parent / "data" / "Data_PE_2025-CSI3_CIR3.xlsx"

df = pd.read_excel(file, index_col=0)  
df = df.dropna(how="all")    
print(df.head())        # contrôle visuel
X = StandardScaler().fit_transform(df.values)

# --- 1) Sélectionner les variables numériques
num_df = df.select_dtypes(include=[np.number])

# --- 2) Imputer les valeurs manquantes (moyenne par défaut)
imputer = SimpleImputer(strategy="mean")
X_num   = imputer.fit_transform(num_df)

# --- 3) Vérifier qu’il ne reste ni NaN ni Inf
assert np.isfinite(X_num).all(), "Encore des valeurs non finies !"

# --- 4) Standardiser
X_std = StandardScaler().fit_transform(X_num)

# --- 5) Linkage Ward
Z = linkage(X_std, method="ward")          # ⚔ plus d’exception

import numpy as np, scipy.spatial as sp
d = sp.distance.pdist(X_std, metric="euclidean")
print("NaN ?", np.isnan(d).any(), "| Inf ?", np.isinf(d).any())

# 1) Choix du nombre de classes (ou coupe à hauteur h)
k = 3
labels = fcluster(Z, k, criterion="maxclust")   # étiquettes CAH
df["cluster"] = labels                         # stocke dans le DataFrame

# 2) Dendrogramme avec ligne de coupe
plt.figure(figsize=(9, 4))
dendrogram(Z, labels=df.index.to_numpy(), color_threshold=Z[-k, 2])
plt.axhline(Z[-k, 2], ls="--", lw=1, c="k")
plt.title(f"Dendrogramme — coupe à k = {k}")
plt.ylabel("∆ inertie intra-classe")
plt.tight_layout();  plt.show()

# 3) Tableau de synthèse (effectif, moyenne, écart-type)
summary = (
    df.groupby("cluster")
      .agg(["count", "mean", "std"])
      .T                                          # transpose pour lisibilité
)
print("\n── Résumé des clusters ──")
print(summary)

# 4) Score silhouette global
silh = silhouette_score(X_std, labels)
print(f"\nSilhouette global : {silh:.3f}")

# 5) Projection PCA 2-D colorée par cluster
pca = PCA(n_components=2, random_state=0)
coords_2d = pca.fit_transform(X_std)
plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=labels,
            cmap="tab10", s=40, alpha=0.8)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Nuage PCA coloré par cluster")
plt.grid(True); plt.tight_layout(); plt.show()

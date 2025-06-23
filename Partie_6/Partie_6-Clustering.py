# Importation des librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, pairwise_distances
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# 1. Charger les données
df = pd.read_excel("Data_PE_2025-CSI3_CIR3.xlsx")
df = df.dropna(how='all')
print(df.head())

# On enlève les colonnes non numériques
df_num = df.select_dtypes(include=[np.number])

# 2. Prétraitement : standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)

# Création d’un dossier pour sauvegarder les figures
os.makedirs("figures", exist_ok=True)

# 3. ACP (PCA) - réduction en 2D et 3D
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', s=50, alpha=0.7)
plt.title("Projection ACP (PCA) - 2 composantes principales")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.savefig("figures/Projection_PCA.png")
plt.close()

explained_variance_ratio = pca_2d.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"Variance expliquée par PC1 : {explained_variance_ratio[0]:.3%}")
print(f"Variance expliquée par PC2 : {explained_variance_ratio[1]:.3%}")
print(f"Variance cumulée sur 2 composantes : {cumulative_variance[1]:.3%}")

print(f"Variance expliquée par PC1 : {pca_3d.explained_variance_ratio_[0]*100:.3f}%")
print(f"Variance expliquée par PC2 : {pca_3d.explained_variance_ratio_[1]*100:.3f}%")
print(f"Variance expliquée par PC3 : {pca_3d.explained_variance_ratio_[2]*100:.3f}%")
print(f"Variance cumulée sur 3 composantes : {np.sum(pca_3d.explained_variance_ratio_)*100:.3f}%")

# 4. t-SNE - visualisation en 2D et 3D
tsne_2d = TSNE(n_components=2, perplexity=10, max_iter=1000, random_state=42)
X_tsne = tsne_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='green', s=50, alpha=0.7)
plt.title("Projection t-SNE (2D)")
plt.grid(True)
plt.savefig("figures/Projection_tSNE.png")
plt.close()

tsne_3d = TSNE(n_components=3, perplexity=10, max_iter=1000, random_state=42)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c='green', s=50, alpha=0.7)
ax.set_title("Projection t-SNE - 3D")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_zlabel("t-SNE 3")
plt.savefig("figures/Projection_tSNE_3D.png")
plt.close()

# 5. Heatmap des distances
distance_matrix = squareform(pdist(X_scaled, metric='euclidean'))

plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, cmap="viridis")
plt.title("Heatmap des distances entre observations")
plt.savefig("figures/Heatmap_Distances.png")
plt.close()

# 6. 3D PCA (visualisation)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c='blue', s=50, alpha=0.7)
ax.set_title("Projection ACP (PCA) - 3D")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
plt.savefig("figures/Projection_PCA_3D.png")
plt.close()

# --- Fonctions clustering et évaluation ---

def is_valid_clustering(labels):
    unique = set(labels)
    if -1 in unique:
        unique.remove(-1)
    return len(unique) > 1

def dunn_index(X, labels):
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2:
        return np.nan
    intra_dists = []
    inter_dists = []
    for c in unique_clusters:
        cluster_points = X[labels == c]
        if len(cluster_points) < 2:
            intra_dists.append(0)
        else:
            intra = pairwise_distances(cluster_points)
            intra_dists.append(np.max(intra))  # diamètre du cluster
    for i, c1 in enumerate(unique_clusters):
        for c2 in unique_clusters[i+1:]:
            dist = pairwise_distances(X[labels == c1], X[labels == c2])
            inter_dists.append(np.min(dist))  # distance minimale entre clusters
    if max(intra_dists) == 0:
        return np.nan
    return np.min(inter_dists) / np.max(intra_dists)

def evaluate_clustering(X, labels, method_name):
    print(f"\n--- Évaluation {method_name} ---")
    if not is_valid_clustering(labels):
        print("Pas assez de clusters valides pour calculer les indices.")
        return None
    try:
        sil = silhouette_score(X, labels)
    except:
        sil = np.nan
    try:
        db = davies_bouldin_score(X, labels)
    except:
        db = np.nan
    dunn = dunn_index(X, labels)
    print(f"Silhouette Score: {sil:.3f}")
    print(f"Davies-Bouldin Index (plus bas meilleur): {db:.3f}")
    print(f"Dunn Index (plus haut meilleur): {dunn:.3f}")
    return sil, db, dunn

# 7. Clustering sur données complètes standardisées
n_clusters = 3

# CAH
labels_cah = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_scaled)
# K-means
labels_kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_scaled)
# DBSCAN
labels_dbscan = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)

evaluate_clustering(X_scaled, labels_cah, "CAH sur données complètes")
evaluate_clustering(X_scaled, labels_kmeans, "K-means sur données complètes")
evaluate_clustering(X_scaled, labels_dbscan, "DBSCAN sur données complètes")

# 8. Clustering sur PCA 2D
labels_cah_pca2d = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_pca)
labels_kmeans_pca2d = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_pca)
labels_dbscan_pca2d = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_pca)

evaluate_clustering(X_pca, labels_cah_pca2d, "CAH sur PCA 2D")
evaluate_clustering(X_pca, labels_kmeans_pca2d, "K-means sur PCA 2D")
evaluate_clustering(X_pca, labels_dbscan_pca2d, "DBSCAN sur PCA 2D")

# 9. Clustering sur PCA 3D
labels_cah_pca3d = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_pca_3d)
labels_kmeans_pca3d = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_pca_3d)
labels_dbscan_pca3d = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_pca_3d)

evaluate_clustering(X_pca_3d, labels_cah_pca3d, "CAH sur PCA 3D")
evaluate_clustering(X_pca_3d, labels_kmeans_pca3d, "K-means sur PCA 3D")
evaluate_clustering(X_pca_3d, labels_dbscan_pca3d, "DBSCAN sur PCA 3D")

# 10. Clustering sur t-SNE 2D
labels_cah_tsne2d = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_tsne)
labels_kmeans_tsne2d = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_tsne)
labels_dbscan_tsne2d = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_tsne)

evaluate_clustering(X_tsne, labels_cah_tsne2d, "CAH sur t-SNE 2D")
evaluate_clustering(X_tsne, labels_kmeans_tsne2d, "K-means sur t-SNE 2D")
evaluate_clustering(X_tsne, labels_dbscan_tsne2d, "DBSCAN sur t-SNE 2D")

# 11. Clustering sur t-SNE 3D
labels_cah_tsne3d = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(X_tsne_3d)
labels_kmeans_tsne3d = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_tsne_3d)
labels_dbscan_tsne3d = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_tsne_3d)

evaluate_clustering(X_tsne_3d, labels_cah_tsne3d, "CAH sur t-SNE 3D")
evaluate_clustering(X_tsne_3d, labels_kmeans_tsne3d, "K-means sur t-SNE 3D")
evaluate_clustering(X_tsne_3d, labels_dbscan_tsne3d, "DBSCAN sur t-SNE 3D")

# --- Recherche du meilleur eps pour DBSCAN sur données complètes ---
eps_values = np.linspace(0.1, 1.5, 15)
best_eps = None
best_score = -1
best_labels = None

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels_tmp = db.fit_predict(X_scaled)
    if is_valid_clustering(labels_tmp):
        try:
            score = silhouette_score(X_scaled, labels_tmp)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_labels = labels_tmp
        except:
            continue

if best_eps is not None:
    print(f"\nMeilleur eps DBSCAN : {best_eps:.2f} avec silhouette score : {best_score:.3f}")
    evaluate_clustering(X_scaled, best_labels, "DBSCAN Optimisé")
else:
    print("\nDBSCAN - Aucun clustering valide trouvé sur la plage eps testée.")

# --- Visualisation clusters ---

def plot_clusters(X_proj, labels, method_name, dim_reduc_name):
    plt.figure(figsize=(8,6))
    unique_labels = np.unique(labels)
    n_clusters_plot = len(unique_labels) - (1 if -1 in labels else 0)
    if n_clusters_plot < 1:
        print(f"{method_name} - Pas assez de clusters valides pour visualisation.")
        return
    palette = sns.color_palette("hsv", n_clusters_plot)
    colors = []
    for lbl in labels:
        if lbl == -1:
            colors.append('lightgrey')
        else:
            colors.append(palette[lbl % n_clusters_plot])
    plt.scatter(X_proj[:,0], X_proj[:,1], c=colors, s=50, alpha=0.7)
    plt.title(f"{method_name} - Clusters sur {dim_reduc_name}")
    plt.xlabel(f"{dim_reduc_name} 1")
    plt.ylabel(f"{dim_reduc_name} 2")
    plt.grid(True)
    plt.savefig(f"figures/Clusters_{method_name.replace(' ', '_')}_{dim_reduc_name}.png")
    plt.close()

# Visualisation clusters sur PCA 2D
plot_clusters(X_pca, labels_cah_pca2d, "CAH", "PCA_2D")
plot_clusters(X_pca, labels_kmeans_pca2d, "K-means", "PCA_2D")
plot_clusters(X_pca, labels_dbscan_pca2d, "DBSCAN", "PCA_2D")

# Visualisation clusters sur t-SNE 2D
plot_clusters(X_tsne, labels_cah_tsne2d, "CAH", "tSNE_2D")
plot_clusters(X_tsne, labels_kmeans_tsne2d, "K-means", "tSNE_2D")
plot_clusters(X_tsne, labels_dbscan_tsne2d, "DBSCAN", "tSNE_2D")

# Visualisation clusters sur données complètes (sur PCA 2D pour visibilité)
plot_clusters(X_pca, labels_cah, "CAH", "Données_Complètes")
plot_clusters(X_pca, labels_kmeans, "K-means", "Données_Complètes")
plot_clusters(X_pca, labels_dbscan, "DBSCAN", "Données_Complètes")
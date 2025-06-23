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
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

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

# 3. ACP (PCA) - réduction en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', s=50, alpha=0.7)
plt.title("Projection ACP (PCA) - 2 composantes principales")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.grid(True)
plt.savefig("figures/Projection_PCA.png")
plt.close()

# 4. t-SNE - visualisation en 2D
tsne = TSNE(n_components=2, perplexity=10, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='green', s=50, alpha=0.7)
plt.title("Projection t-SNE (2D)")
plt.grid(True)
plt.savefig("figures/Projection_tSNE.png")
plt.close()

# 5. Heatmap des distances
distance_matrix = squareform(pdist(X_scaled, metric='euclidean'))

plt.figure(figsize=(10, 8))
sns.heatmap(distance_matrix, cmap="viridis")
plt.title("Heatmap des distances entre observations")
plt.savefig("figures/Heatmap_Distances.png")
plt.close()

# ----------------------
# 6. Clustering et indices d’évaluation
# ----------------------

# Méthode 1 : CAH (Clustering Hiérarchique Ascendant)
n_clusters = 3  # à adapter
cah = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
labels_cah = cah.fit_predict(X_scaled)

# Méthode 2 : K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Méthode 3 : DBSCAN (exemple)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)

# Indices d’évaluation pour CAH
silhouette_cah = silhouette_score(X_scaled, labels_cah)
# Dunn index n’est pas dans sklearn, on peut approximer avec Davies-Bouldin (inverse)
db_index_cah = davies_bouldin_score(X_scaled, labels_cah)

print(f"CAH - Silhouette Score: {silhouette_cah:.3f}")
print(f"CAH - Davies-Bouldin Index (plus bas meilleur): {db_index_cah:.3f}")

# Indices pour K-means
silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)
db_index_kmeans = davies_bouldin_score(X_scaled, labels_kmeans)

print(f"K-means - Silhouette Score: {silhouette_kmeans:.3f}")
print(f"K-means - Davies-Bouldin Index: {db_index_kmeans:.3f}")

# Indices pour DBSCAN (si au moins 2 clusters)
if len(set(labels_dbscan)) > 1 and -1 not in set(labels_dbscan):
    silhouette_dbscan = silhouette_score(X_scaled, labels_dbscan)
    db_index_dbscan = davies_bouldin_score(X_scaled, labels_dbscan)
    print(f"DBSCAN - Silhouette Score: {silhouette_dbscan:.3f}")
    print(f"DBSCAN - Davies-Bouldin Index: {db_index_dbscan:.3f}")
else:
    print("DBSCAN - Pas assez de clusters pour calculer les indices.")

# ----------------------
# 7. Validation croisée / stabilité
# ----------------------

# Simple validation croisée via split train-test (pour CAH et K-means)
X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

cah_train = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(X_train)
labels_train = cah_train.labels_

# Affecter labels_test par prédiction kNN nearest centroid (pour stabilité)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, labels_train)
labels_test_pred = knn.predict(X_test)

# Clustering complet sur test pour comparaison
cah_test = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit(X_test)
labels_test = cah_test.labels_

# Calcul Adjusted Rand Index entre labels test et prédits (stabilité)
ari = adjusted_rand_score(labels_test, labels_test_pred)
print(f"Validation croisée (ARI entre prédiction et clustering sur test) : {ari:.3f}")

# ----------------------
# 8. Statistiques descriptives par cluster (exemple CAH)
# ----------------------

df_clusters = df_num.copy()
df_clusters['cluster'] = labels_cah

print("\nStatistiques descriptives par cluster (moyenne et écart-type):")
print(df_clusters.groupby('cluster').agg(['mean', 'std']))

# ----------------------
# 9. Analyse critique
# ----------------------

print("\n--- Analyse critique ---")
print("• Le choix du critère de distance et de la méthode linkage (ici 'ward') impacte fortement le clustering.")
print("• Les données ont été standardisées pour limiter l’effet d’échelle.")
print("• Les méthodes comme DBSCAN sont plus adaptées pour détecter des clusters de forme arbitraire et résistent mieux aux outliers.")
print("• Une réduction de dimension via ACP a été effectuée pour visualisation mais le clustering s’est fait sur les données complètes standardisées.")
print("• Pour une meilleure robustesse, il est conseillé de tester plusieurs méthodes et indices d’évaluation.")
print("• La validation croisée montre la stabilité relative du clustering, mais peut être améliorée par d’autres techniques.")
print("• L’interprétation des clusters via statistiques descriptives aide à donner un sens aux groupes détectés.")

print("\n✅ Toutes les figures ont été sauvegardées dans le dossier 'figures'")





from sklearn.metrics import pairwise_distances

# --- 1. Visualisation clusters colorés sur PCA et t-SNE ---

def plot_clusters(X_proj, labels, method_name, dim_reduc_name):
    plt.figure(figsize=(8,6))
    palette = sns.color_palette("hsv", len(np.unique(labels)))
    sns.scatterplot(x=X_proj[:,0], y=X_proj[:,1], hue=labels, palette=palette, legend='full', s=50)
    plt.title(f"{method_name} clusters projetés via {dim_reduc_name}")
    plt.xlabel(f"{dim_reduc_name} 1")
    plt.ylabel(f"{dim_reduc_name} 2")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.savefig(f"figures/{method_name}_{dim_reduc_name}_clusters.png")
    plt.close()

# Visualisation pour CAH
plot_clusters(X_pca, labels_cah, "CAH", "PCA")
plot_clusters(X_tsne, labels_cah, "CAH", "tSNE")

# Visualisation pour K-means
plot_clusters(X_pca, labels_kmeans, "Kmeans", "PCA")
plot_clusters(X_tsne, labels_kmeans, "Kmeans", "tSNE")

# Visualisation pour DBSCAN (si clusters valides)
if len(set(labels_dbscan)) > 1 and -1 not in set(labels_dbscan):
    plot_clusters(X_pca, labels_dbscan, "DBSCAN", "PCA")
    plot_clusters(X_tsne, labels_dbscan, "DBSCAN", "tSNE")

# --- 2. Calcul de l’indice de Dunn ---

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
    return np.min(inter_dists) / np.max(intra_dists)

print(f"Dunn index CAH : {dunn_index(X_scaled, labels_cah):.3f}")
print(f"Dunn index K-means : {dunn_index(X_scaled, labels_kmeans):.3f}")
if len(set(labels_dbscan)) > 1 and -1 not in set(labels_dbscan):
    print(f"Dunn index DBSCAN : {dunn_index(X_scaled, labels_dbscan):.3f}")
else:
    print("DBSCAN - pas assez de clusters valides pour Dunn index.")

# --- 3. Recherche du meilleur eps pour DBSCAN (silhouette) ---

eps_values = np.linspace(0.1, 1.5, 15)
best_eps = None
best_score = -1
best_labels = None

for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=5)
    labels_tmp = db.fit_predict(X_scaled)
    # Silhouette possible seulement si au moins 2 clusters sans outliers (-1)
    n_clusters_tmp = len(set(labels_tmp)) - (1 if -1 in labels_tmp else 0)
    if n_clusters_tmp > 1:
        try:
            score = silhouette_score(X_scaled, labels_tmp)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_labels = labels_tmp
        except:
            pass

if best_eps is not None:
    print(f"Meilleur eps DBSCAN : {best_eps:.2f} avec silhouette score : {best_score:.3f}")
    # Visualiser les clusters DBSCAN optimaux
    plot_clusters(X_pca, best_labels, "DBSCAN_Optim", "PCA")
    plot_clusters(X_tsne, best_labels, "DBSCAN_Optim", "tSNE")
else:
    print("DBSCAN - Aucun clustering valide trouvé sur la plage eps testée.")
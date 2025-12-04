# 04 — Validación de Clustering vs Ratings (ARI y NMI)

# En este notebook:

#- Cargamos indicadores escalados y ranking procesado.
#- Realizamos K-Means (k=5) sobre t-SNE o directamente sobre X.
#- Construimos `df_cluster_labels`.
#- Emparejamos cooperativas y ratings.
#- Calculamos ARI y NMI.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import unicodedata, re
from rapidfuzz import process, fuzz

df_indicadores_financieros_scaled_trasp = pd.read_csv(
    "df_indicadores_financieros_scaled_trasp.csv",
    index_col=0
)
df_ranking_ordenado = pd.read_csv("df_ranking_ordenado.csv")

# ======================================
# 1. t-SNE para visualización (opcional) y K-Means
# ======================================

X = df_indicadores_financieros_scaled_trasp.values

tsne = TSNE(n_components=2, random_state=42, perplexity=10)
X_tsne = tsne.fit_transform(X)

k_opt = 5
kmeans = KMeans(n_clusters=k_opt, random_state=0, n_init=50)
labels_K = kmeans.fit_predict(X_tsne)

df_clusters_km = df_indicadores_financieros_scaled_trasp.copy()
df_clusters_km["cluster_km"] = labels_K

df_clusters_km.to_csv("df_clusters_km.csv")


# ======================================
# 2. Construir df_cluster_labels
# ======================================

df_cluster_labels = df_clusters_km.reset_index().rename(
    columns={"index": "Cooperativa", "cluster_km": "Cluster"}
)

df_cluster_labels = df_cluster_labels[["Cooperativa", "Cluster"]]
display(df_cluster_labels.head())

# ======================================
# 3. Normalizar claves y hacer matching con RapidFuzz
# ======================================

def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    texto = re.sub(r'[^a-z0-9]+', '_', texto)
    texto = texto.strip('_')
    return texto

df_cluster_labels["Cooperativa_Key"] = df_cluster_labels["Cooperativa"].apply(normalizar)
df_ranking_ordenado["Institucion_Key"] = df_ranking_ordenado["Institucion"].apply(normalizar)

cluster_keys = df_cluster_labels["Cooperativa_Key"].tolist()
rating_keys = df_ranking_ordenado["Institucion_Key"].tolist()

matches = []
for r in rating_keys:
    best_match, score, _ = process.extractOne(
        r,
        cluster_keys,
        scorer=fuzz.partial_ratio
    )
    matches.append({
        "rating_key": r,
        "best_cluster_key": best_match,
        "score": score
    })

df_matches = pd.DataFrame(matches).sort_values(by="score", ascending=False)
df_good = df_matches[df_matches["score"] >= 90].copy()
map_dict = dict(zip(df_good["rating_key"], df_good["best_cluster_key"]))

df_ranking_ordenado["Cooperativa_Key_Traducida"] = df_ranking_ordenado["Institucion_Key"].map(map_dict)

# ======================================
# 4. Construir df_validacion y calcular ARI/NMI
# ======================================

df_validacion = pd.merge(
    df_cluster_labels,
    df_ranking_ordenado,
    left_on="Cooperativa_Key",
    right_on="Cooperativa_Key_Traducida",
    how="inner"
)

print("Filas emparejadas:", len(df_validacion))
display(df_validacion[["Cooperativa", "Cluster", "rating_num"]].head())

ari = adjusted_rand_score(df_validacion["Cluster"], df_validacion["rating_num"])
nmi = normalized_mutual_info_score(df_validacion["Cluster"], df_validacion["rating_num"])

print("Adjusted Rand Index (ARI):", ari)
print("Normalized Mutual Information (NMI):", nmi)

df_validacion.to_csv("df_validacion.csv", index=False)

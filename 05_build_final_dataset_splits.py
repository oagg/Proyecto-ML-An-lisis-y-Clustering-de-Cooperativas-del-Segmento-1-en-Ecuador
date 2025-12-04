# 05 — Dataset final y generación de splits estratificados

#En este notebook:

#- Construimos X_final y y_final usando indicadores + ratings.
#- Analizamos la distribución de clases.
#- Generamos splits estratificados para p ∈ {0.05, 0.10, 0.20, 0.40, 0.60, 0.80}.


import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df_indicadores_financieros_scaled_trasp = pd.read_csv(
    "df_indicadores_financieros_scaled_trasp.csv",
    index_col=0
)
df_validacion = pd.read_csv("df_validacion.csv")

# ======================================
# 1. X_full e y_full a partir de df_validacion
# ======================================

X_full = df_indicadores_financieros_scaled_trasp.copy()
df_labels = df_validacion[["Cooperativa", "rating_num"]].drop_duplicates()

df_final = X_full.merge(
    df_labels,
    left_index=True,
    right_on="Cooperativa",
    how="inner"
)

df_final = df_final.set_index("Cooperativa")
X_final = df_final.drop(columns=["rating_num"])
y_final = df_final["rating_num"].astype(int)

print("Shape X_final:", X_final.shape)
print("Shape y_final:", y_final.shape)
print("Distribución de clases:")
print(y_final.value_counts().sort_index())

# ======================================
# 2. Función para generar splits estratificados
# ======================================

def generar_splits_corregidos(
    X,
    y,
    p_list=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80],
    reps=10
):
    splits = {}
    N = len(X)
    n_classes = y.nunique()

    for p in p_list:
        labeled_target = int(p * N)
        if labeled_target < n_classes:
            labeled_target = n_classes

        frac_labeled = labeled_target / N
        splits[p] = []

        for seed in range(reps):
            sss1 = StratifiedShuffleSplit(
                n_splits=1,
                train_size=frac_labeled,
                random_state=seed
            )

            for labeled_idx, rest_idx in sss1.split(X, y):
                rest_all = rest_idx.copy()
                np.random.seed(seed)
                np.random.shuffle(rest_all)

                mitad = len(rest_all) // 2
                unlabeled_idx = rest_all[:mitad]
                test_idx = rest_all[mitad:]

                splits[p].append({
                    "labeled_idx": labeled_idx,
                    "unlabeled_idx": unlabeled_idx,
                    "test_idx": test_idx,
                    "labeled_X": X.iloc[labeled_idx],
                    "labeled_y": y.iloc[labeled_idx],
                    "unlabeled_X": X.iloc[unlabeled_idx],
                    "test_X": X.iloc[test_idx],
                    "test_y": y.iloc[test_idx]
                })
    return splits

# ======================================
# 3. Generar splits y revisar distribución
# ======================================

splits_corregidos = generar_splits_corregidos(X_final, y_final)

for p in splits_corregidos:
    primera = splits_corregidos[p][0]
    print(f"p={p} → labeled_y counts:", primera["labeled_y"].value_counts().to_dict())

# 06 — Baseline supervisado: Random Forest

#En este notebook:

#- Usamos los splits estratificados.
#- Entrenamos un RandomForestClassifier solo con labeled_X.
#- Evaluamos sobre test_X.
#- Calculamos Macro-F1 y Balanced Accuracy.
#- Resumimos el rendimiento promedio por p.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

# Debes volver a cargar X_final, y_final y regenerar splits_corregidos (o importarlos del notebook anterior)
df_indicadores_financieros_scaled_trasp = pd.read_csv(
    "df_indicadores_financieros_scaled_trasp.csv",
    index_col=0
)
df_validacion = pd.read_csv("df_validacion.csv")

X_full = df_indicadores_financieros_scaled_trasp.copy()
df_labels = df_validacion[["Cooperativa", "rating_num"]].drop_duplicates()
df_final = X_full.merge(df_labels, left_index=True, right_on="Cooperativa", how="inner")
df_final = df_final.set_index("Cooperativa")
X_final = df_final.drop(columns=["rating_num"])
y_final = df_final["rating_num"].astype(int)

# Reutilizamos la función de splits del notebook anterior

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

splits_corregidos = generar_splits_corregidos(X_final, y_final)

# ======================================
# 1. Funciones de métricas y baseline RF
# ======================================

def compute_metrics(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred)
    }

def baseline_rf_one_split(labeled_X, labeled_y, test_X, test_y, seed=42):
    clf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=seed
    )
    clf.fit(labeled_X, labeled_y)
    preds = clf.predict(test_X)
    return compute_metrics(test_y, preds)

# ======================================
# 2. Ejecutar baseline sobre todos los splits
# ======================================

resultados_baseline = []

for p, lista_splits in splits_corregidos.items():
    for rep, split in enumerate(lista_splits):
        metrics = baseline_rf_one_split(
            split["labeled_X"],
            split["labeled_y"],
            split["test_X"],
            split["test_y"],
            seed=42 + rep
        )
        metrics.update({
            "p": p,
            "rep": rep,
            "method": "baseline_rf"
        })
        resultados_baseline.append(metrics)

df_results_baseline = pd.DataFrame(resultados_baseline)
display(df_results_baseline.head())
df_results_baseline.to_csv("df_results_baseline.csv", index=False)

# ======================================
# 3. Resumen por p
# ======================================

resumen_baseline = (
    df_results_baseline
    .groupby("p")[["macro_f1", "balanced_acc"]]
    .agg(["mean", "std"])
    .reset_index()
)

print("📊 Resumen Baseline RF por fracción etiquetada p:")
display(resumen_baseline)

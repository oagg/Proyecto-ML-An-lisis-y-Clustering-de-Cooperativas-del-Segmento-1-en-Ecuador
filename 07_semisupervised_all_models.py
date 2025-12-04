# 07 — Modelos semisupervisados: Self-Training, Label Spreading, Label Propagation

#En este notebook:

#- Reutilizamos X_final, y_final y splits_corregidos.
#- Ejecutamos Self-Training con diferentes umbrales τ.
#- Ejecutamos Label Spreading con diferentes gammas.
#- Ejecutamos Label Propagation con diferentes k vecinos.
#- Resumimos resultados y comparamos con el baseline.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Reconstruir X_final, y_final y splits (igual que antes)
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

from sklearn.model_selection import StratifiedShuffleSplit

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
# 1. Self-Training
# ======================================

def compute_metrics(y_true, y_pred):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred)
    }

def self_training_rf_one_split(split, tau, seed=42):
    labeled_X = split["labeled_X"]
    labeled_y = split["labeled_y"]
    unlabeled_X = split["unlabeled_X"]
    test_X = split["test_X"]
    test_y = split["test_y"]

    clf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=seed
    )
    clf.fit(labeled_X, labeled_y)

    probs_unl = clf.predict_proba(unlabeled_X)
    max_probs = probs_unl.max(axis=1)
    pseudo_mask = max_probs >= tau

    X_pseudo = unlabeled_X[pseudo_mask]
    y_pseudo = clf.predict(unlabeled_X[pseudo_mask])

    X_aug = pd.concat([labeled_X, X_pseudo], axis=0)
    y_aug = pd.concat([labeled_y, pd.Series(y_pseudo, index=X_pseudo.index)])

    clf_aug = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=seed
    )
    clf_aug.fit(X_aug, y_aug)

    preds_test = clf_aug.predict(test_X)

    metrics = compute_metrics(test_y, preds_test)
    metrics.update({
        "pseudo_labels": int(pseudo_mask.sum())
    })
    return metrics

resultados_st = []

for p, lista_splits in splits_corregidos.items():
    for rep, split in enumerate(lista_splits):
        for tau in [0.6, 0.7, 0.8, 0.9]:
            metrics = self_training_rf_one_split(split, tau, seed=42+rep)
            metrics.update({
                "p": p,
                "rep": rep,
                "tau": tau,
                "method": f"self_training_tau_{tau}"
            })
            resultados_st.append(metrics)

df_self_training = pd.DataFrame(resultados_st)
df_self_training.to_csv("df_self_training.csv", index=False)
display(df_self_training.head())

# ======================================
# 2. Label Spreading
# ======================================

def run_label_spreading_experiments(X, y, splits, reps=10):
    rows = []
    for p, lista_splits in splits.items():
        for rep, split in enumerate(lista_splits):
            labeled_X = split["labeled_X"]
            labeled_y = split["labeled_y"]
            unlabeled_X = split["unlabeled_X"]
            test_X = split["test_X"]
            test_y = split["test_y"]

            X_all = pd.concat([labeled_X, unlabeled_X, test_X], axis=0)
            y_all = -1 * np.ones(len(X_all), dtype=int)
            y_all[: len(labeled_y)] = labeled_y.values

            for gamma in [0.1, 0.25, 0.5, 1.0]:
                model = LabelSpreading(
                    kernel="rbf",
                    gamma=gamma,
                    max_iter=30
                )
                model.fit(X_all.values, y_all)

                # Predicciones solo para test
                start_test = len(labeled_y) + len(unlabeled_X)
                y_pred_test = model.transduction_[start_test:]

                metrics = compute_metrics(test_y, y_pred_test)
                metrics.update({
                    "p": p,
                    "rep": rep,
                    "gamma": gamma,
                    "method": "label_spreading"
                })
                rows.append(metrics)
    return pd.DataFrame(rows)

df_ls = run_label_spreading_experiments(X_final, y_final, splits_corregidos)
df_ls.to_csv("df_label_spreading.csv", index=False)
display(df_ls.head())

# ======================================
# 3. Label Propagation
# ======================================

def run_label_propagation_experiments(X, y, splits, reps=10):
    rows = []
    for p, lista_splits in splits.items():
        for rep, split in enumerate(lista_splits):
            labeled_X = split["labeled_X"]
            labeled_y = split["labeled_y"]
            unlabeled_X = split["unlabeled_X"]
            test_X = split["test_X"]
            test_y = split["test_y"]

            X_all = pd.concat([labeled_X, unlabeled_X, test_X], axis=0)
            y_all = -1 * np.ones(len(X_all), dtype=int)
            y_all[: len(labeled_y)] = labeled_y.values

            for n_neighbors in [5, 10, 20]:
                model = LabelPropagation(
                    kernel="knn",
                    n_neighbors=n_neighbors,
                    max_iter=30
                )
                model.fit(X_all.values, y_all)
                start_test = len(labeled_y) + len(unlabeled_X)
                y_pred_test = model.transduction_[start_test:]

                metrics = compute_metrics(test_y, y_pred_test)
                metrics.update({
                    "p": p,
                    "rep": rep,
                    "neighbors": n_neighbors,
                    "method": "label_propagation"
                })
                rows.append(metrics)
    return pd.DataFrame(rows)

df_lp = run_label_propagation_experiments(X_final, y_final, splits_corregidos)
df_lp.to_csv("df_label_propagation.csv", index=False)
display(df_lp.head())

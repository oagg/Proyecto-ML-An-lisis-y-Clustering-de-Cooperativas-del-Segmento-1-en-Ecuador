# 03 — Normalización de nombres y matching ratings-cooperativas

#En este notebook:

#- Limpiamos el DataFrame `df_ranking_raw` del PDF.
#- Filtramos Segmento 1.
#- Creamos una columna `rating_num`.
#- Normalizamos nombres para poder compararlos con los de indicadores.
#- Guardamos un ranking procesado listo para emparejar.

import numpy as np
import pandas as pd
import re
import unicodedata
from rapidfuzz import process, fuzz

df_ranking_raw = pd.read_csv("df_ranking_raw.csv")
df_indicadores_financieros_scaled_trasp = pd.read_csv(
    "df_indicadores_financieros_scaled_trasp.csv",
    index_col=0
)

# ======================================
# 1. Limpieza básica de df_ranking_raw
# ======================================

df_ranking = df_ranking_raw.dropna(how="all").copy()
df_ranking = df_ranking.applymap(lambda x: str(x).replace("\n", " ").strip() if pd.notna(x) else x)

display(df_ranking.head())

# ======================================
# 2. Ajustar encabezados
# ======================================

header = df_ranking.iloc[0].tolist()
df_ranking = df_ranking[1:].copy()
df_ranking.columns = header

df_ranking.columns = [c.replace("\r", " ").replace("  ", " ").strip() for c in df_ranking.columns]

rename_map = {
    "N": "No.",
    "RUC": "RUC",
    "INSTITUCIÓN FINANCIERA": "Institucion",
    "SEGMENTO": "Segmento",
    "FIRMA CALIFICADORA DE RIESGO": "Firma",
    "AL 30 DE JUNIO 2024": "Jun2024",
    "AL 30 DE SEPTIEMBRE 2024": "Sep2024",
    "AL 31 DE DICIEMBRE 2024": "Dic2024",
    "AL 31 DE MARZO 2025": "Mar2025",
    "AL 30 DE JUNIO 2025": "Jun2025"
}
df_ranking = df_ranking.rename(columns=rename_map)

display(df_ranking.head())

# ======================================
# 3. Normalizar ratings, filtrar Segmento 1 y crear rating_num
# ======================================

def limpiar(s):
    if pd.isna(s):
        return None
    s = str(s)
    s = s.replace("–", "-").replace("—", "-")
    s = s.upper()
    s = s.replace("´", "'")
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

df_ranking = df_ranking.applymap(limpiar)

# Normalizar Segmento
df_ranking["Segmento"] = df_ranking["Segmento"].astype(str).str.extract(r'(\d+)')[0]
df_ranking["Segmento"] = df_ranking["Segmento"].astype(float).astype("Int64")

# Filtrar Segmento 1
df_ranking_seg1 = df_ranking[df_ranking["Segmento"] == 1].copy()

# ======================================
# 4. Quedarse solo con la última evaluación y crear rating_num
# ======================================

ultima_columna = df_ranking_seg1.columns[-1]

rating_scale = {
    "AAA": 1, "AAA-": 1,
    "AA+": 2, "AA": 2, "AA-": 2,
    "A+": 3, "A": 3, "A-": 3,
    "BBB+": 4, "BBB": 4, "BBB-": 4,
    "BB+": 5, "BB": 5, "BB-": 5,
    "B+": 5, "B": 5, "B-": 5,
}

def convertir_rating(valor, escala):
    if pd.isna(valor):
        return None
    valor = str(valor).replace("\r", " ").replace("\n", " ").strip()
    valor = re.sub(r"\s*[/|]\s*", "|", valor)
    valor = re.sub(r"\s+", " ", valor)
    if "|" not in valor:
        return escala.get(valor, None)
    partes = [p.strip() for p in valor.split("|")]
    valores = [escala.get(p, None) for p in partes]
    valores = [v for v in valores if v is not None]
    if not valores:
        return None
    return sum(valores) / len(valores)

df_ranking_seg1["rating_num"] = df_ranking_seg1[ultima_columna].apply(
    lambda x: convertir_rating(x, rating_scale)
)

df_ranking_seg1 = df_ranking_seg1[df_ranking_seg1["rating_num"].notna()].copy()

display(df_ranking_seg1[["Institucion", "rating_num"]].head())
df_ranking_seg1.to_csv("df_ranking_ordenado.csv", index=False)

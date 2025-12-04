# 02 — Limpieza y escalado de indicadores financieros

#En este notebook:

#- Cargamos la hoja de indicadores financieros desde el XLSM de Segmento 1.
#- Limpiamos índices y columnas.
#- Convertimos porcentajes a valores decimales.
#- Estandarizamos los indicadores (StandardScaler).
#- Generamos `df_indicadores_financieros_scaled_trasp`.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unidecode
from sklearn.preprocessing import StandardScaler

# Recuperar ruta_segmento1 guardada en el notebook 1
ruta_segmento1 = np.load("ruta_segmento1.npy", allow_pickle=True)[0]
print("Usando archivo Segmento 1:", ruta_segmento1)

# ======================================
# 1. Cargar hoja de INDICADORES FINANCIEROS
# ======================================

df_indicadores_financieros = pd.read_excel(
    ruta_segmento1,
    sheet_name="5. INDICADORES FINANCIEROS",
    header=16
)

print("Hoja '5. INDICADORES FINANCIEROS' cargada:")
display(df_indicadores_financieros.head())

# ======================================
# 2. Ajustar índice y eliminar filas NaN
# ======================================

# Usar la primera columna como índice (nombres de indicadores)
df_indicadores_financieros = df_indicadores_financieros.set_index(
    df_indicadores_financieros.columns[0]
)

# Quitar filas completamente vacías
df_indicadores_financieros = df_indicadores_financieros.dropna(how="all")

print("Después de set_index y dropna:")
display(df_indicadores_financieros.head())

# ======================================
# 3. Normalizar nombres (índices y columnas)
# ======================================

def normalizar_texto(valor):
    if not isinstance(valor, str):
        return valor
    valor = valor.lower()
    valor = unidecode.unidecode(valor)
    valor = re.sub(r'[^a-z0-9]+', '_', valor)
    valor = re.sub(r'_+', '_', valor)
    valor = valor.strip('_')
    return valor

df_indicadores_financieros.index = df_indicadores_financieros.index.map(normalizar_texto)
df_indicadores_financieros.columns = df_indicadores_financieros.columns.map(normalizar_texto)

print("Normalización completada para índices y columnas:")
display(df_indicadores_financieros.head())

# ======================================
# 4. Convertir porcentajes a decimales y escalar
# ======================================

df_fixed = df_indicadores_financieros.copy()

# Quitar el símbolo %
df_fixed = df_fixed.replace('%', '', regex=True)

# Convertir todo a número
df_fixed = df_fixed.apply(pd.to_numeric, errors='coerce')

# Pasar de 100% → 1.0
df_fixed = df_fixed / 100

# Llenar NaN con 0 (o podrías usar otra estrategia)
df_fixed = df_fixed.fillna(0)

print("Indicadores convertidos a decimales (ejemplo):")
display(df_fixed.iloc[:5, :5])

# ======================================
# 5. Escalado con StandardScaler
# ======================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_fixed)

df_indicadores_financieros_scaled = pd.DataFrame(
    X_scaled,
    index=df_fixed.index,
    columns=df_fixed.columns
)

# Transponer: filas = cooperativas, columnas = indicadores
df_indicadores_financieros_scaled_trasp = df_indicadores_financieros_scaled.T

print("Dimensiones finales (cooperativas x indicadores):", df_indicadores_financieros_scaled_trasp.shape)
display(df_indicadores_financieros_scaled_trasp.head())

# Guardar para notebooks siguientes
df_indicadores_financieros_scaled_trasp.to_csv("df_indicadores_financieros_scaled_trasp.csv")

# 01 — Descarga y extracción de datos SEPS
#- Descargamos el ZIP oficial de SEPS con las hojas de Excel (Segmento 1).
#- Extraemos el archivo de Segmento 1 (XLSM).
#- Descargamos el PDF con las calificaciones de riesgo.
#- Extraemos las tablas del PDF con `tabula-py`.
#

# ======================================
# 1. Librerías e instalación inicial
# ======================================
# !pip install tabula-py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import os
import zipfile
from datetime import datetime
import tabula
import re

# ======================================
# 2. Parámetros de descarga
# ======================================

URL_DESCARGA = "https://estadisticas.seps.gob.ec/?sdm_process_download=1&download_id=2776"
CARPETA_BASE = "seps_descargas"
CARPETA_SEG1 = "seps_segmento1"

os.makedirs(CARPETA_BASE, exist_ok=True)
os.makedirs(CARPETA_SEG1, exist_ok=True)
# ======================================
# 3. Funciones utilitarias
# ======================================

def descargar_archivo_zip():
    fecha = datetime.now().strftime("%Y%m%d")
    ruta_archivo = f"{CARPETA_BASE}/seps_{fecha}.zip"

    print("Descargando archivo ZIP de SEPS...")
    r = requests.get(URL_DESCARGA)
    r.raise_for_status()

    with open(ruta_archivo, "wb") as f:
        f.write(r.content)

    print(f"Archivo descargado: {ruta_archivo}")
    return ruta_archivo


def extraer_segmento1(ruta_archivo):
    fecha = datetime.now().strftime("%Y%m%d")
    carpeta_temp = f"{CARPETA_BASE}/extraido_{fecha}"
    os.makedirs(carpeta_temp, exist_ok=True)

    with zipfile.ZipFile(ruta_archivo, "r") as z:
        z.extractall(carpeta_temp)

    print("Archivos extraídos. Buscando archivo Segmento 1...")

    archivo_segmento1 = None
    for root, dirs, files in os.walk(carpeta_temp):
        for file in files:
            if (
                "segmento" in file.lower()
                and "1" in file.lower()
                and file.endswith((".xls", ".xlsx", ".xlsm"))
            ):
                archivo_segmento1 = os.path.join(root, file)

    if archivo_segmento1 is None:
        raise FileNotFoundError("No se encontró el archivo Segmento 1 dentro del ZIP.")

    destino = os.path.join(CARPETA_SEG1, f"{fecha}_segmento1.xlsm")
    with open(archivo_segmento1, "rb") as f_origen, open(destino, "wb") as f_destino:
        f_destino.write(f_origen.read())

    print(f"Archivo Segmento 1 listo: {destino}")
    return destino
# ======================================
# 4. Ejecutar descarga y extracción
# ======================================

zipfile_path = descargar_archivo_zip()
ruta_segmento1 = extraer_segmento1(zipfile_path)

print("Ruta final Segmento 1:", ruta_segmento1)
# ======================================
# 5. Descargar PDF de Calificación de Riesgo
# ======================================

URL_RATING = "https://www.seps.gob.ec/?sdm_process_download=1&download_id=28370"
NOMBRE_PDF = "Calificacion_de_Riesgo.pdf"

print(f"Descargando PDF desde:\n{URL_RATING}\n")
resp = requests.get(URL_RATING)
resp.raise_for_status()

with open(NOMBRE_PDF, "wb") as f:
    f.write(resp.content)

print(f"PDF guardado como: {NOMBRE_PDF}")

# ======================================
# 6. Extraer tablas del PDF de ratings
# ======================================

pdf_path = NOMBRE_PDF

tablas = tabula.read_pdf(pdf_path, pages="all", lattice=True, multiple_tables=True)
df_ranking_raw = pd.concat(tablas, ignore_index=True)

print("Primeras filas crudas del PDF:")
display(df_ranking_raw.head())

# Guardar versión cruda por si acaso
df_ranking_raw.to_csv("df_ranking_raw.csv", index=False)
np.save("ruta_segmento1.npy", np.array([ruta_segmento1], dtype=object))

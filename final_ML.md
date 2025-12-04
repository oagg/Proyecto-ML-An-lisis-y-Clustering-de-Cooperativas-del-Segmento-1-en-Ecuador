**Proyecto Final de Machine Learning**
======================================

**Aprendizaje Supervisado y Semisupervisado para la Clasificación de Ratings Financieros de Cooperativas**
----------------------------------------------------------------------------------------------------------

Este proyecto implementa un pipeline completo para:

1.  **Descargar, extraer y procesar datos reales de la SEPS (Ecuador)**
    
2.  **Extraer indicadores financieros desde PDFs oficiales**
    
3.  **Integrar el archivo de ratings oficiales**
    
4.  **Normalizar y alinear nombres de cooperativas**
    
5.  **Realizar clustering no supervisado**
    
6.  **Comparar clusters vs ratings (ARI, NMI)**
    
7.  **Preparar un dataset consistente para clasificación**
    
8.  **Generar splits estratificados para aprendizaje semisupervisado**
    
9.  **Entrenar modelos bajo distintos niveles de etiquetas:**
    
    *   Baseline supervisado (Random Forest)
        
    *   Self-Training
        
    *   Label Spreading
        
    *   Label Propagation
        
10.  **Analizar resultados y comparar métodos**
    

El objetivo final es **evaluar si los patrones financieros pueden predecir ratings oficiales** y cómo el aprendizaje semisupervisado ayuda en escenarios con pocas etiquetas.

📁 **Estructura del Proyecto**
==============================

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   /ML_Project  │  ├── data/  │   ├── indicadores/                  # Tablas extraídas SEPS (PDF → CSV)  │   ├── ratings/                      # Archivo oficial de ratings  │   ├── merged_dataset.csv            # Dataset final  │  ├── notebooks/  │   ├── 01_descarga_y_extraccion.ipynb  │   ├── 02_normalizacion_y_matching.ipynb  │   ├── 03_clustering_y_metricas.ipynb  │   ├── 04_supervised_baseline.ipynb  │   ├── 05_semisupervised_selftraining.ipynb  │   ├── 06_label_spreading_propagation.ipynb  │  ├── results/  │   ├── df_validacion.csv  │   ├── df_results_baseline.csv  │   ├── df_self_training.csv  │   ├── df_label_spreading.csv  │   ├── df_label_propagation.csv  │   ├── plots/  │       ├── resumen_baseline.png  │       ├── resumen_st.png  │       ├── comparación_modelos.png  │  └── README.md   ← este archivo   `

🧩 **1\. Requisitos**
=====================

### 🌐 Ejecutar en Google Colab (recomendado)

Todo funciona out-of-the-box.

### 📦 Librerías necesarias

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install pandas numpy seaborn matplotlib scikit-learn rapidfuzz tabula-py  pip install jpype1       # Necesario para tabula   `

🚀 **2\. Cómo ejecutar el proyecto**
====================================

**PASO 1 — Descarga automática de datos SEPS**
----------------------------------------------

Ejecuta:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from datetime import datetime  import requests, zipfile, os  URL = "https://estadisticas.seps.gob.ec/?sdm_process_download=1&download_id=2776"  fecha = datetime.now().strftime("%Y%m%d")  os.makedirs("seps_descargas", exist_ok=True)  ruta = f"seps_descargas/seps_{fecha}.zip"  r = requests.get(URL, stream=True)  open(ruta, "wb").write(r.content)  with zipfile.ZipFile(ruta, "r") as z:      z.extractall("seps_segmento1")   `

Esto produce tablas PDF que contienen todos los indicadores del Segmento 1.

**PASO 2 — Extracción de indicadores desde PDF**
------------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   import tabula  tablas = tabula.read_pdf("archivo.pdf", pages="all", multiple_tables=True)   `

Se combinan las tablas en un único DataFrame (df\_indicadores).

**PASO 3 — Cargar ratings oficiales**
-------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   df_ranking = pd.read_excel("ratings.xlsx")   `

**PASO 4 — Normalización y Matching de Cooperativas**
-----------------------------------------------------

Usamos RapidFuzz + normalización propia:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from rapidfuzz import process, fuzz  def normalizar(texto):      texto = texto.lower()      texto = unicodedata.normalize("NFKD", texto).encode("ascii","ignore").decode()      texto = re.sub(r'[^a-z0-9]+', '_', texto)      return texto.strip('_')   `

Se obtuvieron **42 coincidencias robustas** (score ≥ 90).

Generaste correctamente df\_validacion con:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Cooperativa — Cluster — rating_num   `

**PASO 5 — Clustering + Métricas ARI y NMI**
--------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  ARI = adjusted_rand_score(df_validacion["Cluster"], df_validacion["rating_num"])  NMI = normalized_mutual_info_score(df_validacion["Cluster"], df_validacion["rating_num"])   `

### Resultado:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ARI = 0.1566  NMI = 0.2703   `

**PASO 6 — Dataset final para aprendizaje**
-------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   X_final = indicadores_finales  y_final = ratings_mapeados   `

Shape final:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   X_final: (42, 41)  y_final: (42,)   `

Distribución de clases:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`1 → 2 instancias    2 → 14    3 → 12    4 → 12    5 → 2`  

**PASO 7 — Splits Estratificados (p = 5–80%)**
----------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   splits_corregidos = generar_splits_corregidos(X_final, y_final)   `

Cada split contiene:

*   labeled\_X, labeled\_y
    
*   unlabeled\_X
    
*   test\_X, test\_y
    

Estratificación correcta garantizada.

**PASO 8 — Baseline Supervisado (Random Forest)**
-------------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   df_results_baseline = ejecutar_baseline_rf(splits_corregidos)   `

Resumen:

*   Rendimiento limitado para p pequeños
    
*   Mejora fuerte desde p ≥ 0.60
    

**PASO 9 — Self-Training**
--------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   df_self_training = ejecutar_self_training(splits_corregidos)   `

Hallazgos:

*   τ=0.6 agrega ruido → baja performance
    
*   τ≥0.8 → muy estable (casi igual al baseline)
    
*   El modelo rara vez mejora sin muchos datos etiquetados
    

**PASO 10 — Label Spreading y Label Propagation**
-------------------------------------------------

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   df_ls = run_label_spreading_experiments(...)  df_lp = run_label_propagation_experiments(...)   `

Comportamiento:

*   Sensible al número de vecinos (k)
    
*   Consistente para p pequeños (similar al baseline)
    
*   Mejora leve sobre Self-Training en p=0.40–0.60
    

**PASO 11 — Visualizaciones Finales**
-------------------------------------

Incluye:

*   Comparación RF vs Self-Training vs LS vs LP
    
*   Efecto de p
    
*   Efecto de τ
    
*   Curvas Macro-F1 y Balanced Accuracy
    

🧠 **Conclusiones principales**
===============================

### 📌 Sobre los datos

*   Dataset pequeño (42 cooperativas)
    
*   Clases fuertemente desbalanceadas
    
*   Ratings tienen ruido y variabilidad entre calificadoras
    

### 📌 Sobre métodos supervisados

*   RF funciona bien con al menos 60% de datos etiquetados
    
*   Macro-F1 máximo ≈ 0.38–0.40 (razonable dado el problema)
    

### 📌 Sobre métodos semisupervisados

*   Self-Training solo ayuda cuando p > 0.40
    
*   Label Spreading/Propagation consistentemente estables
    
*   Mejoran ligeramente en escenarios intermedios
    

### 📌 Conclusión general

Los métodos semisupervisados pueden ayudar, pero **su desempeño está limitado por el tamaño y balance del dataset**, y por la complejidad de los ratings reales.

🔧 **Cómo reproducir exactamente todo**
=======================================

### Opción 1 — Google Colab (recomendado)

1.  https://colab.research.google.com/drive/1mJCdONqHgF9Edd3eJPH_v4bDjux4LKNv?usp=sharing
    


👤 **Autor**
============

Gabriel Avalos
Omar Gordillo

Universidad San Francisco de Quito

Curso: Machine Learning
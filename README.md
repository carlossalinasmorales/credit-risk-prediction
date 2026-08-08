# Credit Risk Prediction

Proyecto de análisis y modelado predictivo para estimar la probabilidad de incumplimiento de pago de una solicitud de crédito. El flujo utiliza el dataset **Home Credit Default Risk**, compara una regresión logística con un Random Forest y exporta el modelo final junto con sus transformadores.

## Inicio rápido

1. Instala [uv](https://docs.astral.sh/uv/) y sincroniza las dependencias:

   ```bash
   uv sync
   ```

2. Coloca el archivo `application_train.csv` en `data/input/`.

3. Inicia Jupyter:

   ```bash
   uv run --with jupyterlab jupyter lab
   ```

4. Abre y ejecuta `notebooks/HomeCredit.ipynb` desde la primera celda hasta el final.

El notebook genera los artefactos en `data/output/modelos/`.

## Objetivo

El problema se formula como una clasificación binaria:

- `TARGET = 0`: pago normal.
- `TARGET = 1`: dificultades de pago o incumplimiento.

El objetivo de negocio es identificar solicitudes de mayor riesgo sin descartar innecesariamente clientes solventes. Debido al desbalance aproximado de las clases, se utiliza `class_weight="balanced"` y se analiza el umbral de decisión en lugar de asumir que `0.5` es siempre óptimo.

## Flujo del análisis

El notebook documenta las siguientes etapas:

1. Carga y exploración del dataset.
2. Renombrado de variables para facilitar la interpretación.
3. Filtrado inicial de columnas de bajo valor analítico o con demasiados valores nulos.
4. Imputación, estandarización y codificación one-hot mediante `ColumnTransformer`.
5. Selección de variables con ANOVA (`SelectKBest`), con `k=40` elegido mediante validación cruzada.
6. Entrenamiento de una regresión logística como baseline.
7. Entrenamiento y ajuste de un `RandomForestClassifier` con `GridSearchCV` y ROC-AUC como métrica.
8. Evaluación con accuracy, precision, recall, F1-score, matriz de confusión y ROC-AUC.
9. Selección de un umbral que maximiza F1-score.
10. Exportación del modelo y de los transformadores con `joblib`.

## Datos

El notebook espera el archivo:

```text
data/input/application_train.csv
```

El dataset contiene 307.511 registros y 122 columnas originales. El análisis reduce la matriz a 36 columnas antes del preprocesamiento y obtiene 147 variables después de la transformación. El archivo de datos no se incluye en Git mediante `.gitignore`; debe obtenerse desde la fuente del dataset de Home Credit y colocarse manualmente en la ruta indicada.

## Resultados registrados

El Random Forest optimizado obtuvo los siguientes hiperparámetros en la búsqueda documentada:

```text
n_estimators: 300
max_depth: 12
min_samples_leaf: 5
ROC-AUC medio en validación cruzada: 0.7415
```

Resultados sobre el conjunto de prueba:

| Modelo | Umbral | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0.5000 | 0.6860 | 0.1585 | 0.6705 | 0.2564 | 0.7426 |
| Random Forest base | 0.5000 | 0.7031 | 0.1635 | 0.6508 | 0.2614 | 0.7427 |
| Random Forest optimizado | 0.5000 | 0.7251 | 0.1720 | 0.6304 | 0.2702 | 0.7449 |
| Random Forest optimizado | 0.5793 | 0.8171 | 0.2118 | 0.4651 | 0.2910 | 0.7449 |

El umbral `0.5793` maximiza el F1-score en el análisis ejecutado. Su uso mejora accuracy, precision y F1 frente al umbral por defecto, pero reduce recall; la elección debe ajustarse al coste de falsos positivos y falsos negativos del negocio.

## Artefactos generados

La última sección del notebook guarda estos archivos en `data/output/modelos/`:

| Archivo | Descripción |
| --- | --- |
| `modelo_home_credit_rf.joblib` | Random Forest optimizado. |
| `preprocessor_home_credit.joblib` | Transformaciones de preparación de datos. |
| `selector_home_credit.joblib` | Selector de las variables elegidas. |
| `umbral_clasificacion.joblib` | Umbral de decisión seleccionado. |

Estos archivos son salidas locales y también están excluidos del control de versiones.

## Estructura

```text
.
├── data/
│   ├── input/
│   │   └── application_train.csv
│   └── output/
│       └── modelos/
├── notebooks/
│   └── HomeCredit.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

## Tecnologías

- Python 3.12 o superior
- pandas y NumPy para manipulación de datos
- scikit-learn para preprocesamiento, selección de variables y modelos
- Matplotlib y Seaborn para visualización
- Joblib para serialización de artefactos
- JupyterLab para ejecutar el análisis

## Limitaciones y siguientes pasos

- El modelo utiliza únicamente `application_train.csv`; no incorpora tablas satélite como `bureau.csv` o `previous_application.csv`.
- Las probabilidades no están calibradas.
- El desbalance de clases limita el recall alcanzable.
- El notebook no expone una API ni un servicio de inferencia.

Como mejoras, se pueden incorporar historiales crediticios adicionales, comparar modelos como XGBoost o LightGBM y calibrar probabilidades con `CalibratedClassifierCV`.

## Nota de uso

Este proyecto tiene fines educativos y analíticos. Los resultados no deben utilizarse como único criterio para aprobar o rechazar solicitudes de crédito sin validación adicional, revisión de sesgos, calibración y controles de cumplimiento normativo.

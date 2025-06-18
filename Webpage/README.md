
# App de Análisis y Predicción de Ocupación Hotelera

Esta aplicación permite realizar **análisis exploratorios (EDA)**, diagnóstico de **calidad de datos**, segmentación de clientes y **predicción de ocupación hotelera** a partir de datos de reservaciones. Utiliza una combinación de **Flask**, **Pandas**, **Azure Blob Storage**, **Chart.js**, y modelos alojados en **Azure Machine Learning**.

---

## Ejecución local

```bash
# 1. Activar el entorno virtual
source my398env/bin/activate

# 2. Ubicarte en la raíz del proyecto
cd /ruta/a/tu/proyecto

# 3. Exportar la variable de entorno para Flask
export FLASK_APP=app.py

# 4. Ejecutar el servidor de desarrollo
flask run
```

> Asegúrate de tener el archivo `.env` configurado con tus credenciales de Azure (`SAS_URL`, `AZURE_SCORING_URI`, `AZURE_API_KEY`).

---

## Infraestructura y Componentes

### Backend
- **Flask**: servidor web y lógica de enrutamiento
- **Flask-Caching**: para almacenamiento temporal de datasets
- **Pandas / NumPy**: procesamiento de datos
- **Azure Blob Storage**: carga de archivos `.csv` desde contenedores protegidos
- **Azure ML**: para predicción de ocupación

### Frontend
- **Jinja2 + HTML5 + Tailwind CSS**: plantillas y estilo visual
- **Chart.js**: visualización de gráficos e insights

---

## Estructura del proyecto

```
src/
└── app/
    ├── templates/          ← Plantillas HTML
    ├── static/             ← Archivos estáticos (JS, CSS)
    ├── data/               ← CSVs y parámetros del modelo
app.py                     ← Archivo principal (Flask App)
.env                       ← Variables de entorno
```

---

## Rutas principales

| Ruta | Descripción |
|------|-------------|
| `/` / `/inicio` | Página de bienvenida |
| `/analisis` | Dashboard de análisis general |
| `/analisis/carga_de_datos` | Carga datos vía Azure o CSV local |
| `/analisis/calidad_y_procesamiento` | Revisión de calidad de datos y metadatos |
| `/modelado` | Visualización de entrenamiento, predicción y forecast |
| `/predict` (POST) | API REST para predicción con modelo Azure ML |
| `/dashboard` | Dashboard interactivo (mockup) |
| `/contacto` | Página de contacto |
| `/download/quality_report` | Descarga del reporte de calidad en HTML |
| `/download/processed_dataset` | Descarga del dataset procesado en CSV |
| `/api/eda/overview` | API con métricas generales del EDA |
| `/api/eda/operational` | API con métricas operativas (canales, tipos, adelantos) |
| `/api/eda/segmentation` | API con segmentación por país, estado, agencia, programa |
| `/api/eda/correlations` | API con heatmap día/mes de ocupación |
| `/api/forecast` | API mockup de predicción futura para dashboard |

---

## Autores

Este proyecto fue desarrollado por:

- **Blas René Treviño Cuéllar** – Ingeniero de Datos  
  ✉️ a01177729@tec.mx | 📞 +52 81 1266 7749

- **Andrés Villarreal González** – Científico de Datos  
  ✉️ a00833915@tec.mx | 📞 +52 833 311 1329

- **Rodrigo González Zermeño** – Ingeniero MLOps  
  ✉️ a00572213@tec.mx | 📞 +52 477 786 6179

- **Héctor Hibran Tapia Fernández** – Desarrollador Full Stack  
  ✉️ a01661114@tec.mx | 📞 +52 56 1551 8463

- **David Antonio Figueroa Campos** – Analista de Datos  
  ✉️ a01198034@tec.mx | 📞 +52 81 1470 7021

# Predicción de Reservas de Hoteles

Este proyecto utiliza técnicas avanzadas de análisis de series de tiempo para predecir la demanda de reservas hoteleras. Combina modelos estadísticos y de machine learning con un despliegue completo en la nube, incluyendo una interfaz web amigable y una API funcional.

## Estructura del Proyecto

```
tca-time-series-forecast/
│
├── Azure-Model-Deployment/       
│   ├── 0_modelo.ipynb
│   ├── 1_entrenador.ipynb
│   ├── 2_deployador.ipynb
│   ├── 3_score.py
│   └── 4_API.ipynb
│
├── Evidence/                     
│   ├── Evidencia 1 - Elaboración de Business Case
│   └── Evidencia 2 - Data Wrangling
│
├── TCA-Webpage/                  
│   ├── src/app/
│   ├── app.py                    
│   ├── requirements.txt
│   ├── pyproject.toml
│   ├── uv.lock
│   └── README.md
│
├── .gitignore
├── .gitattributes
└── README.md
```

## Tecnologías Utilizadas

### Backend & Modelado
- **Python** (Pandas, NumPy, Scikit-learn, Statsmodels)
- **Azure ML** para entrenamiento, scoring y despliegue del modelo
- **Flask** para construcción de la API

### Frontend
- **HTML** para estructura de la página
- **CSS** para estilos personalizados
- **JavaScript** para interactividad y consumo de API

### DevOps
- **Git & GitHub** para control de versiones
- **Azure** como plataforma de despliegue en la nube

## Funcionalidades

- Limpieza y preparación de datos históricos
- Predicción de demanda hotelera a futuro
- Evaluación automática de modelos
- Despliegue del modelo como API REST en Azure
- Interfaz web para interacción con el usuario
- Visualización de predicciones y métricas clave

## Equipo de Desarrollo

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

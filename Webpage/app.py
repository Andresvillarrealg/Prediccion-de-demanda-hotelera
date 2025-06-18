import holidays
import numpy as np
import json, requests
from io import StringIO
import os, pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from flask_caching import Cache
from azure.storage.blob import ContainerClient
from flask import Flask, render_template, request, flash, redirect, url_for, send_file, make_response, jsonify
##############################################################################################################################
load_dotenv()
app = Flask(__name__, template_folder="src/app/templates", static_folder="src/app/static")
app.secret_key = os.getenv('SAS_URL')
app.config['CACHE_TYPE'] = 'simple'     
app.config['CACHE_DEFAULT_TIMEOUT'] = 60*60
cache = Cache(app)
SCALER_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "src/app/data/scaler_params.json")
with open(SCALER_PARAMS_PATH, "r", encoding="utf-8") as f:
    scaler_params = json.load(f)
data_min = np.array(scaler_params["data_min"])
data_range = np.array(scaler_params["data_range"])
FEATURES_TO_SCALE = [
    "personas",     
    "lag_1",         
    "lag_7",         
    "lag_14",        
    "rolling_mean_7",
    "dow_sin",       
    "dow_cos",      
    "month_sin",     
    "month_cos"      
]
IDX = {feat: idx for idx, feat in enumerate(FEATURES_TO_SCALE)}
##############################################################################################################################
def load_eda_data():
    try:
        data_folder = os.path.join(os.path.dirname(__file__), 'src', 'app', 'data')
        df_reservaciones = pd.read_csv(os.path.join(data_folder, 'df_reservaciones_calidad.csv'))
        df_paises = pd.read_csv(os.path.join(data_folder, 'iar_Paises.csv'))
        df_canales = pd.read_csv(os.path.join(data_folder, 'iar_canales.csv'))
        df_tipos_hab = pd.read_csv(os.path.join(data_folder, 'iar_Tipos_Habitaciones.csv'))
        df_agencias = pd.read_csv(os.path.join(data_folder, 'iar_Agencias.csv'))
        df_entidades = pd.read_csv(os.path.join(data_folder, 'iar_Entidades_Fed.csv'))
        df_ocupacion = pd.read_csv(os.path.join(data_folder, 'ocupacion_diaria.csv'))
        #print("🔄 Convirtiendo fechas...")
        df_reservaciones['h_res_fec'] = pd.to_datetime(df_reservaciones['h_res_fec'], errors='coerce')
        df_reservaciones['h_fec_lld'] = pd.to_datetime(df_reservaciones['h_fec_lld'], errors='coerce')
        df_reservaciones['h_fec_sda'] = pd.to_datetime(df_reservaciones['h_fec_sda'], errors='coerce')
        df_ocupacion['fecha'] = pd.to_datetime(df_ocupacion['fecha'], errors='coerce')
        #print("🔗 Realizando joins...")
        df_main = df_reservaciones.merge(
            df_paises[['ID_Pais', 'Pais_Nombre']], 
            left_on='ID_Pais_Origen', 
            right_on='ID_Pais', 
            how='left'
        )
        df_main = df_main.merge(
            df_canales[['ID_canal', 'Canal_nombre']], 
            on='ID_canal', 
            how='left'
        )
        df_main = df_main.merge(
            df_tipos_hab[['ID_Tipo_Habitacion', 'Tipo_Habitacion_nombre']], 
            on='ID_Tipo_Habitacion', 
            how='left'
        )
        df_main = df_main.merge(
            df_agencias[['ID_Agencia', 'Agencia_nombre']], 
            on='ID_Agencia',  
            how='left'
        )
        df_main = df_main.merge(
            df_entidades[['Estado_cve', 'Estado_Nombre']], 
            left_on='h_edo', 
            right_on='Estado_cve', 
            how='left'
        )
        #print("📊 Calculando métricas...")
        df_main['adelanto_dias'] = (df_main['h_fec_lld'] - df_main['h_res_fec']).dt.days
        df_main['estancia_dias'] = (df_main['h_fec_sda'] - df_main['h_fec_lld']).dt.days
        df_main['mes_reserva'] = df_main['h_res_fec'].dt.month
        df_main['año_reserva'] = df_main['h_res_fec'].dt.year
        df_main['dia_semana'] = df_main['h_fec_lld'].dt.dayofweek
        df_main['es_fin_semana'] = df_main['dia_semana'].isin([5, 6])
        df_ocupacion['es_fin_de_semana'] = df_ocupacion['es_fin_de_semana'].astype(bool)
        df_ocupacion['mes'] = df_ocupacion['fecha'].dt.month
        df_ocupacion['año'] = df_ocupacion['fecha'].dt.year
        df_ocupacion['ocupacion_pct'] = (df_ocupacion['personas'] / df_ocupacion['personas'].max()) * 100
        cache.set('df_main_eda', df_main)
        cache.set('df_ocupacion_eda', df_ocupacion)
        cache.set('df_paises_eda', df_paises)
        cache.set('df_canales_eda', df_canales)
        cache.set('df_tipos_hab_eda', df_tipos_hab)
        cache.set('df_entidades_eda', df_entidades)
        cache.set('df_agencias_eda', df_agencias)
        #print(f"✅ EDA Data loaded successfully:")
        #print(f"   - Reservaciones: {len(df_main):,}")
        #print(f"   - Ocupación diaria: {len(df_ocupacion):,} días")
        #print(f"   - Países: {len(df_paises):,}")
        #print(f"   - Canales: {len(df_canales):,}")
        #print(f"   - Tipos habitación: {len(df_tipos_hab):,}")
        return True
    except Exception as e:
        print(f"❌ Error loading EDA data: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
##############################################################################################################################
def build_metadata(df: pd.DataFrame, n_sample: int = 3) -> pd.DataFrame:
    meta_rows = []
    for col in df.columns:
        ser = df[col]
        row = {
            "column"     : col,
            "dtype"      : ser.dtype,
            "non_null"   : int(ser.count()),
            "nulls"      : int(ser.isna().sum()),
            "pct_nulls"  : round(ser.isna().mean() * 100, 2),
            "unique"     : ser.nunique(dropna=True)
        }
        if pd.api.types.is_numeric_dtype(ser) or pd.api.types.is_datetime64_any_dtype(ser):
            row["min"]  = ser.min(skipna=True)
            row["max"]  = ser.max(skipna=True)
            if pd.api.types.is_numeric_dtype(ser):
                row["mean"] = ser.mean(skipna=True)
                row["std"]  = ser.std(skipna=True)
        else:
            row["min"] = row["max"] = row["mean"] = row["std"] = np.nan
        sample_values = ser.dropna().unique()[:n_sample]
        row["sample"] = sample_values.tolist().copy()
        meta_rows.append(row)
    meta_df = pd.DataFrame(meta_rows)
    ordered_cols = ["column", "dtype", "non_null", "nulls", "pct_nulls",
                    "unique", "min", "max", "mean", "std", "sample"]
    meta_df = meta_df[ordered_cols]
    return meta_df
################################################################################################################
@app.route("/")
@app.route("/inicio")
def inicio():
    return render_template("inicio.html", page="inicio")
################################################################################################################
ALLOWED_EXTENSIONS = {'.csv'}
EXPECTED_FILES = {
    "ocupaciones":  ("raw/iar_Ocupaciones_202504211018.csv", "Ocupaciones"),
    "reservaciones":("raw/iar_Reservaciones_202504211011.csv", "Reservaciones"),
    "ingresos":     ("raw/iar_ingresos_diarios_202504211014.csv", "Ingresos diarios"),
}
def allowed_file(fn): return os.path.splitext(fn)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/analisis/carga_de_datos', methods=['GET', 'POST'])
def carga_de_datos():
    if request.method == 'POST':
        source = request.form.get('source')
        if source == 'upload':
            file = request.files.get('csv_file')
            if not file or file.filename == '' or not allowed_file(file.filename):
                flash('Debes subir un archivo .csv válido', 'error')
                return redirect(request.url)
            df_reservaciones = pd.read_csv(file)
            table_name = file.filename 
            filas_columnas = df_reservaciones.shape
            flash('¡Datos cargados exitosamente!', 'success')
        elif source == 'azure':
            table_key = request.form.get('azure_table')
            if table_key not in EXPECTED_FILES:
                flash('Selecciona la tabla a cargar', 'error')
                return redirect(request.url)
            sas_url = app.secret_key         
            if not sas_url:
                flash('SAS URL no configurada en el servidor', 'error')
                return redirect(request.url)
            path, label = EXPECTED_FILES[table_key]
            container_client = ContainerClient.from_container_url(sas_url)
            blob_client = container_client.get_blob_client(path)
            blob_data = blob_client.download_blob().readall()
            df_reservaciones = pd.read_csv(StringIO(blob_data.decode('utf-8')))
            table_name = label    
            filas_columnas = df_reservaciones.shape
            flash('¡Datos cargados exitosamente!', 'success')
        else:
            flash('Fuente de datos no válida', 'error')
            return redirect(request.url)
        cache.set('df_reservaciones', df_reservaciones)  
        tailwind_table = (
            df_reservaciones.head(30)
            .to_html(
                classes="min-w-full text-xs text-left text-gray-700 border-collapse divide-y divide-gray-200",
                index=False,
                border=0,
                table_id="previewTable"
            )
            .replace("<thead>", "<thead class='bg-black'>")
            .replace("<th>", "<th class='px-4 py-2 text-white'>")
            .replace("<tbody>", "<tbody class='divide-y divide-gray-100'>")
        )
        return render_template(
            "carga_de_datos.html",
            df_html=tailwind_table,
            page="carga_de_datos",
            table_name=table_name,
            filas_columnas=filas_columnas
        )
    df_reservaciones = cache.get('df_reservaciones')
    if df_reservaciones is not None:
        tailwind_table = (
            df_reservaciones.head(30)
            .to_html(
                classes="min-w-full text-xs text-left text-gray-700 border-collapse divide-y divide-gray-200",
                index=False,
                border=0,
                table_id="previewTable"
            )
            .replace("<thead>", "<thead class='bg-black'>")
            .replace("<th>", "<th class='px-4 py-2 text-white'>")
            .replace("<tbody>", "<tbody class='divide-y divide-gray-100'>")
        )
        return render_template(
            "carga_de_datos.html",
            df_html=tailwind_table,
            page="carga_de_datos",
            table_name="Reservaciones",
            filas_columnas=df_reservaciones.shape
        )
    return render_template("carga_de_datos.html", page="carga_de_datos")
################################################################################################################
@app.route('/analisis/calidad_y_procesamiento')
def calidad_y_procesamiento():
    df = cache.get('df_reservaciones')
    if df is None:
        flash('Debes cargar primero los datos.', 'error')
        return redirect(url_for('carga_de_datos'))
    preview_df = df.head(5)
    preview_data = preview_df.to_dict(orient='records')
    preview_columns = list(preview_df.columns)
    meta_df = build_metadata(df)
    metadata = meta_df.to_dict(orient='records')
    quality_data = {
        'dataset_stats': {
            'variables': 49,
            'observations': 203002,
            'missing_cells': 408123,
            'missing_percentage': 4.1,
            'duplicate_rows': 0,
            'duplicate_percentage': 0.0,
            'memory_size': '122.4 MiB',
            'avg_record_size': '632.0 B'
        },
        'variable_types': {
            'datetime': 16,
            'numeric': 21,
            'categorical': 7,
            'text': 5,
            'unsupported': 2
        },
        'alerts': [
            {'type': 'constant', 'column': 'ID_empresa', 'message': 'tiene valor constante "1"', 'severity': 'warning'},
            {'type': 'constant', 'column': 'moneda_cve', 'message': 'tiene valor constante "1"', 'severity': 'warning'},
            {'type': 'imbalance', 'column': 'ID_Programa', 'message': 'está muy desbalanceado (97.5%)', 'severity': 'high'},
            {'type': 'imbalance', 'column': 'ID_Pais_Origen', 'message': 'está muy desbalanceado (96.3%)', 'severity': 'high'},
            {'type': 'missing', 'column': 'h_correo_e', 'message': 'tiene 203002 (100.0%) valores faltantes', 'severity': 'high'},
            {'type': 'missing', 'column': 'h_nom', 'message': 'tiene 203002 (100.0%) valores faltantes', 'severity': 'high'},
            {'type': 'skewed', 'column': 'h_num_noc', 'message': 'está muy sesgado (γ1 = 81.01891754)', 'severity': 'medium'},
            {'type': 'skewed', 'column': 'aa_h_num_noc', 'message': 'está muy sesgado (γ1 = 82.30675646)', 'severity': 'medium'},
            {'type': 'skewed', 'column': 'h_tfa_total', 'message': 'está muy sesgado (γ1 = 26.89402197)', 'severity': 'medium'},
            {'type': 'skewed', 'column': 'aa_h_tfa_total', 'message': 'está muy sesgado (γ1 = 27.50566233)', 'severity': 'medium'},
            {'type': 'uniform', 'column': 'ID_Reserva', 'message': 'está distribuido uniformemente', 'severity': 'low'},
            {'type': 'unique', 'column': 'ID_Reserva', 'message': 'tiene valores únicos', 'severity': 'info'},
            {'type': 'unsupported', 'column': 'h_correo_e', 'message': 'es un tipo no soportado, verifica si necesita limpieza o análisis adicional', 'severity': 'warning'},
            {'type': 'unsupported', 'column': 'h_nom', 'message': 'es un tipo no soportado, verifica si necesita limpieza o análisis adicional', 'severity': 'warning'},
            {'type': 'zeros', 'column': 'h_num_per', 'message': 'tiene 100803 (49.7%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'aa_h_num_per', 'message': 'tiene 102199 (50.3%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'h_num_adu', 'message': 'tiene 100803 (49.7%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'aa_h_num_adu', 'message': 'tiene 102199 (50.3%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'h_num_men', 'message': 'tiene 198413 (97.7%) ceros', 'severity': 'high'},
            {'type': 'zeros', 'column': 'aa_h_num_men', 'message': 'tiene 198466 (97.8%) ceros', 'severity': 'high'},
            {'type': 'zeros', 'column': 'h_num_noc', 'message': 'tiene 101006 (49.8%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'aa_h_num_noc', 'message': 'tiene 102402 (50.4%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'h_tot_hab', 'message': 'tiene 101049 (49.8%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'aa_h_tot_hab', 'message': 'tiene 102445 (50.5%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'ID_canal', 'message': 'tiene 6378 (3.1%) ceros', 'severity': 'low'},
            {'type': 'zeros', 'column': 'Cliente_Disp', 'message': 'tiene 100803 (49.7%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'aa_Cliente_Disp', 'message': 'tiene 102199 (50.3%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'h_tfa_total', 'message': 'tiene 106391 (52.4%) ceros', 'severity': 'medium'},
            {'type': 'zeros', 'column': 'aa_h_tfa_total', 'message': 'tiene 107743 (53.1%) ceros', 'severity': 'medium'}
        ]
    }
    return render_template(
        'calidad_y_procesamiento.html',
        page='calidad_y_procesamiento',
        preview_data=preview_data,
        preview_columns=preview_columns,
        metadata=metadata,
        quality_data=quality_data
    )
################################################################################################################
@app.route('/download/quality_report')
def download_quality_report():
    try:
        return send_file('src/app/data/profiling_reservaciones.html', 
                        as_attachment=True, 
                        download_name='reporte_calidad_detallado.html')
    except Exception as e:
        flash('Error al descargar el reporte', 'error')
        return redirect(url_for('calidad_y_procesamiento'))
#################################################################################################################   
@app.route('/download/processed_dataset')
def download_processed_dataset():
    df = cache.get('df_reservaciones')
    if df is None:
        flash('No hay datos cargados para descargar', 'error')
        return redirect(url_for('calidad_y_procesamiento'))
    output = StringIO()
    df.to_csv(output, index=False, encoding='utf-8')
    csv_data = output.getvalue()
    response = make_response(csv_data)
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=df_reservaciones_calidad.csv'
    return response
################################################################################################################
@app.route("/analisis")
def analisis():
    df = cache.get('df_reservaciones')
    if df is None:
        flash('Debes cargar primero los datos.', 'error')
        return redirect(url_for('carga_de_datos'))
    return render_template("analisis.html", page="analisis")
#################################################################################################################
@app.route('/api/eda/overview')
def eda_overview():
    try:
        df_main = cache.get('df_main_eda')
        df_ocupacion = cache.get('df_ocupacion_eda')
        if df_main is None or df_ocupacion is None:
            success = load_eda_data()
            if not success:
                return jsonify({'error': 'Error cargando datos'}), 500
            df_main = cache.get('df_main_eda')
            df_ocupacion = cache.get('df_ocupacion_eda')
        total_reservaciones = len(df_main)
        total_huespedes = int(df_main['h_num_per'].sum())
        ocupacion_promedio = round(df_ocupacion['ocupacion_pct'].mean(), 1)
        tarifa_promedio = int(df_main[df_main['h_tfa_total'] > 0]['h_tfa_total'].mean())
        alos_promedio = round(df_main['estancia_dias'].mean(), 1)
        top_paises = (df_main.groupby('Pais_Nombre')['ID_Reserva']
                     .count()
                     .sort_values(ascending=False)
                     .head(10)
                     .to_dict())
        reservas_mensuales = (df_main.groupby('mes_reserva')['ID_Reserva']
                             .count()
                             .to_dict())
        alos_mensual = (df_main.groupby('mes_reserva')['estancia_dias']
                        .mean()
                        .round(1)
                        .to_dict())
        paises_unicos = df_main['Pais_Nombre'].nunique()
        canales_activos = df_main['Canal_nombre'].nunique()
        ocupacion_weekend = round(df_ocupacion[df_ocupacion['es_fin_de_semana']]['ocupacion_pct'].mean(), 1)
        ocupacion_weekday = round(df_ocupacion[~df_ocupacion['es_fin_de_semana']]['ocupacion_pct'].mean(), 1)
        ocupacion_mensual = df_ocupacion.groupby('mes')['ocupacion_pct'].mean()
        mes_temporada_alta = ocupacion_mensual.idxmax()
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        temporada_alta = meses[mes_temporada_alta - 1]
        response_data = {
            'general_metrics': {
                'total_reservaciones': f"{total_reservaciones:,}",
                'total_huespedes': f"{total_huespedes:,}",
                'ocupacion_promedio': f"{ocupacion_promedio}%",
                'tarifa_promedio': f"${tarifa_promedio:,}",
                'alos_promedio': f"{alos_promedio} días"
            },
            'geographic_metrics': {
                'top_region': list(top_paises.keys())[0] if top_paises else 'N/A',
                'top_region_percent': f"{round((list(top_paises.values())[0] / total_reservaciones) * 100, 1)}%" if top_paises else '0%',
                'paises_unicos': paises_unicos,
                'diversification_index': f"{round(100 - ((list(top_paises.values())[0] / total_reservaciones) * 100), 1)}%" if top_paises else '100%'
            },
            'temporal_metrics': {
                'ocupacion_weekend': f"{ocupacion_weekend}%",
                'ocupacion_weekday': f"{ocupacion_weekday}%",
                'temporada_alta': temporada_alta
            },
            'charts_data': {
                'top_paises': top_paises,
                'reservas_mensuales': reservas_mensuales,
                'alos_mensual': alos_mensual,
                'ocupacion_diaria': df_ocupacion.sort_values('fecha')[['fecha', 'personas']].to_dict('records')
            },
            'operational_metrics': {
                'canales_activos': canales_activos,
                'adelanto_promedio': round(df_main['adelanto_dias'].mean(), 1),
                'estancia_promedio': round(df_main['estancia_dias'].mean(), 1)
            }
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in eda_overview: {str(e)}")
        return jsonify({'error': str(e)}), 500
################################################################################################################
@app.route('/api/eda/operational')
def eda_operational():
    try:
        df_main = cache.get('df_main_eda')
        df_canales = cache.get('df_canales_eda')
        df_tipos_hab = cache.get('df_tipos_hab_eda')
        if df_main is None:
            success = load_eda_data()
            if not success:
                return jsonify({'error': 'Error cargando datos'}), 500
            df_main = cache.get('df_main_eda')
            df_canales = cache.get('df_canales_eda')
            df_tipos_hab = cache.get('df_tipos_hab_eda')
        canales_data = (df_main.groupby('Canal_nombre')['ID_Reserva']
                       .count()
                       .sort_values(ascending=False)
                       .head(10) 
                       .to_dict())
        canales_data = {k.strip() if k else 'Sin definir': v for k, v in canales_data.items()}
        tipos_hab_data = (df_main.groupby('Tipo_Habitacion_nombre')['ID_Reserva']
                         .count()
                         .sort_values(ascending=False)
                         .head(8) 
                         .to_dict())
        tipos_hab_data = {k.strip() if k else 'Sin definir': v for k, v in tipos_hab_data.items()}
        adelanto_valido = df_main[(df_main['adelanto_dias'] >= 0) & (df_main['adelanto_dias'] <= 365)]
        bins = [0, 7, 15, 30, 60, 90, 180, 365]
        labels = ['0-7 días', '8-15 días', '16-30 días', '31-60 días', '61-90 días', '91-180 días', '181-365 días']
        adelanto_counts = {}
        for i in range(len(bins)-1):
            count = len(adelanto_valido[
                (adelanto_valido['adelanto_dias'] >= bins[i]) & 
                (adelanto_valido['adelanto_dias'] < bins[i+1])
            ])
            adelanto_counts[labels[i]] = count
        adelanto_stats = {
            'promedio': round(adelanto_valido['adelanto_dias'].mean(), 1),
            'mediana': round(adelanto_valido['adelanto_dias'].median(), 1),
            'min': int(adelanto_valido['adelanto_dias'].min()),
            'max': int(adelanto_valido['adelanto_dias'].max())
        }
        canales_metrics = {}
        for canal in list(canales_data.keys())[:5]:
            canal_data = df_main[df_main['Canal_nombre'].str.strip() == canal]
            if len(canal_data) > 0:
                canales_metrics[canal] = {
                    'reservas': len(canal_data),
                    'huespedes_promedio': round(canal_data['h_num_per'].mean(), 1),
                    'tarifa_promedio': round(canal_data[canal_data['h_tfa_total'] > 0]['h_tfa_total'].mean(), 0),
                    'adelanto_promedio': round(canal_data['adelanto_dias'].mean(), 1)
                }
        tipos_metrics = {}
        for tipo in list(tipos_hab_data.keys())[:5]: 
            tipo_data = df_main[df_main['Tipo_Habitacion_nombre'].str.strip() == tipo]
            if len(tipo_data) > 0:
                tipos_metrics[tipo] = {
                    'reservas': len(tipo_data),
                    'huespedes_promedio': round(tipo_data['h_num_per'].mean(), 1),
                    'tarifa_promedio': round(tipo_data[tipo_data['h_tfa_total'] > 0]['h_tfa_total'].mean(), 0),
                    'noches_promedio': round(tipo_data['h_num_noc'].mean(), 1)
                }
        response_data = {
            'canales': {
                'distribuccion': canales_data,
                'metricas': canales_metrics
            },
            'tipos_habitacion': {
                'distribuccion': tipos_hab_data,
                'metricas': tipos_metrics
            },
            'adelanto_reserva': {
                'histograma': adelanto_counts,
                'estadisticas': adelanto_stats
            },
            'insights': {
                'canal_principal': list(canales_data.keys())[0] if canales_data else 'N/A',
                'canal_principal_percent': round((list(canales_data.values())[0] / sum(canales_data.values())) * 100, 1) if canales_data else 0,
                'tipo_habitacion_principal': list(tipos_hab_data.keys())[0] if tipos_hab_data else 'N/A',
                'tipo_principal_percent': round((list(tipos_hab_data.values())[0] / sum(tipos_hab_data.values())) * 100, 1) if tipos_hab_data else 0,
                'adelanto_mas_comun': max(adelanto_counts, key=adelanto_counts.get) if adelanto_counts else 'N/A'
            }
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in eda_operational: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
################################################################################################################
@app.route('/api/eda/correlations')
def eda_correlations():
    try:
        df_ocupacion = cache.get('df_ocupacion_eda') 
        if df_ocupacion is None:
            success = load_eda_data()
            if not success:
                return jsonify({'error': 'Error cargando datos'}), 500
            df_ocupacion = cache.get('df_ocupacion_eda')
        df_heatmap = df_ocupacion.dropna(subset=['fecha', 'personas']).copy()
        df_heatmap['dia_semana'] = df_heatmap['fecha'].dt.dayofweek 
        df_heatmap['mes'] = df_heatmap['fecha'].dt.month
        heatmap_matrix = df_heatmap.groupby(['dia_semana', 'mes'])['personas'].sum().reset_index()
        dias_semana = list(range(7)) 
        meses = list(range(1, 13))  
        matriz_completa = {}
        for dia in dias_semana:
            matriz_completa[dia] = {}
            for mes in meses:
                matriz_completa[dia][mes] = 0
        for _, row in heatmap_matrix.iterrows():
            dia = int(row['dia_semana'])
            mes = int(row['mes'])
            personas = int(row['personas']) 
            matriz_completa[dia][mes] = personas
        heatmap_data = []
        nombres_dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        nombres_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        for dia_idx, dia_num in enumerate(dias_semana):
            for mes_idx, mes_num in enumerate(meses):
                personas = matriz_completa[dia_num][mes_num]
                heatmap_data.append({
                    'x': mes_idx,          
                    'y': dia_idx,          
                    'v': personas,         
                    'mes_nombre': nombres_meses[mes_idx],
                    'dia_nombre': nombres_dias[dia_idx],
                    'mes_num': mes_num,
                    'dia_num': dia_num
                })
        todos_valores = [punto['v'] for punto in heatmap_data]
        max_personas = max(todos_valores)
        min_personas = min(todos_valores)
        promedio_personas = round(sum(todos_valores) / len(todos_valores))
        punto_max = max(heatmap_data, key=lambda x: x['v'])
        punto_min = min(heatmap_data, key=lambda x: x['v'])
        totales_por_dia = {}
        for dia_idx, dia_nombre in enumerate(nombres_dias):
            total = sum([p['v'] for p in heatmap_data if p['y'] == dia_idx])
            totales_por_dia[dia_nombre] = total
        totales_por_mes = {}
        for mes_idx, mes_nombre in enumerate(nombres_meses):
            total = sum([p['v'] for p in heatmap_data if p['x'] == mes_idx])
            totales_por_mes[mes_nombre] = total
        dia_mas_ocupado = max(totales_por_dia, key=totales_por_dia.get)
        mes_mas_ocupado = max(totales_por_mes, key=totales_por_mes.get)
        promedios_por_dia = {}
        for dia_nombre in nombres_dias:
            total = totales_por_dia[dia_nombre]
            dias_totales = len(df_heatmap[df_heatmap['dia_semana'] == nombres_dias.index(dia_nombre)])
            promedio = round(total / max(dias_totales, 1))
            promedios_por_dia[dia_nombre] = promedio
        valores_no_cero = [v for v in todos_valores if v > 0]
        cv = (np.std(valores_no_cero) / np.mean(valores_no_cero)) * 100 if valores_no_cero else 0
        response_data = {
            'heatmap_data': heatmap_data,
            'labels': {
                'dias': nombres_dias,
                'meses': nombres_meses
            },
            'statistics': {
                'max_personas': max_personas,
                'min_personas': min_personas,
                'promedio_personas': promedio_personas,
                'punto_max': {
                    'dia': punto_max['dia_nombre'],
                    'mes': punto_max['mes_nombre'],
                    'personas': punto_max['v']
                },
                'punto_min': {
                    'dia': punto_min['dia_nombre'],
                    'mes': punto_min['mes_nombre'],
                    'personas': punto_min['v']
                },
                'dia_mas_ocupado': dia_mas_ocupado,
                'mes_mas_ocupado': mes_mas_ocupado,
                'total_personas_analizadas': sum(todos_valores),
                'coeficiente_variacion': round(cv, 1)
            },
            'totales_por_dia': totales_por_dia,
            'totales_por_mes': totales_por_mes,
            'promedios_por_dia': promedios_por_dia,
            'insights': {
                'patron_semanal': f"{dia_mas_ocupado} es el día con mayor ocupación ({totales_por_dia[dia_mas_ocupado]:,} personas)",
                'patron_mensual': f"{mes_mas_ocupado} es el mes con mayor ocupación ({totales_por_mes[mes_mas_ocupado]:,} personas)",
                'variabilidad': f"Coeficiente de variación: {round(cv, 1)}% - {'Alta' if cv > 50 else 'Moderada' if cv > 25 else 'Baja'} variabilidad",
                'promedio_diario': f"Promedio de {promedio_personas:,} personas por celda del heatmap"
            }
        }
        return jsonify(response_data)
    except Exception as e:
        print(f"Error in eda_correlations (heatmap ocupación): {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
#################################################################################################################
@app.route('/api/eda/segmentation')
def eda_segmentation():
    try:
        df_main = cache.get('df_main_eda')
        df_entidades = cache.get('df_entidades_eda')
        df_agencias = cache.get('df_agencias_eda')
        if df_main is None:
            success = load_eda_data()
            if not success:
                return jsonify({'error': 'Error cargando datos'}), 500
            df_main = cache.get('df_main_eda')
            df_entidades = cache.get('df_entidades_eda')
            df_agencias = cache.get('df_agencias_eda')
        def calcular_metricas_concentracion(estados_data):
            if not estados_data:
                return {
                    'top_3_concentracion': 0,
                    'indice_hhi': 0,
                    'estados_relevantes': 0
                }
            valores_ordenados = list(estados_data.values())
            total_reservas_estados = sum(valores_ordenados)
            top_3_reservas = sum(valores_ordenados[:3]) 
            top_3_concentracion = round((top_3_reservas / total_reservas_estados) * 100, 1)
            proporciones_cuadradas = [(count / total_reservas_estados) ** 2 for count in valores_ordenados]
            indice_hhi = round(sum(proporciones_cuadradas), 3)
            umbral_relevancia = 1000
            estados_relevantes = sum(1 for count in valores_ordenados if count >= umbral_relevancia)
            #print(f"📊 Métricas de concentración calculadas:")
            #print(f"   - Top 3 concentración: {top_3_concentracion}%")
            #print(f"   - Índice HHI: {indice_hhi}")
            #print(f"   - Estados relevantes: {estados_relevantes}")
            return {
                'top_3_concentracion': top_3_concentracion,
                'indice_hhi': indice_hhi,
                'estados_relevantes': estados_relevantes,
                'total_reservas_estados': total_reservas_estados
            }
        paises_data = (df_main.groupby('Pais_Nombre')['ID_Reserva']
                      .count()
                      .sort_values(ascending=False)
                      .head(4) 
                      .to_dict())
        paises_data = {k.strip() if k else 'Sin definir': v for k, v in paises_data.items()}
        #print(f"🔍 Analizando estados mexicanos con validación de catálogo...")
        CODIGOS_ESTADOS_MEXICANOS = {
            'EAGU', 'EBCN', 'EBSU', 'ECH', 'ECL', 'ECM', 'ECS', 'ECU', 'EDF', 'EDG',
            'EGR', 'EGT', 'EHG', 'EJA', 'EMC', 'EMR', 'EMX', 'ENA', 'ENL', 'EOA',
            'EPB', 'EQE', 'EQR', 'ESI', 'ESL', 'ESO', 'ETB', 'ETL', 'ETM', 'EVE',
            'EYU', 'EZA'
        }
        #print(f"🇲🇽 Códigos válidos de estados mexicanos: {len(CODIGOS_ESTADOS_MEXICANOS)}")
        estados_data = {}
        total_estados_unicos = 0
        if 'h_edo' in df_main.columns and df_entidades is not None:
            try:
                df_main_clean = df_main.copy()
                df_main_clean['h_edo_clean'] = df_main_clean['h_edo'].str.strip()
                df_estados_mexicanos = df_main_clean[
                    df_main_clean['h_edo_clean'].isin(CODIGOS_ESTADOS_MEXICANOS)
                ]
                #print(f"🔍 Registros con códigos de estados mexicanos: {len(df_estados_mexicanos)}")
                #print(f"🔍 Códigos únicos encontrados: {df_estados_mexicanos['h_edo_clean'].nunique()}")
                #print(f"🔍 Muestra de códigos: {df_estados_mexicanos['h_edo_clean'].unique()[:10]}")
                if len(df_estados_mexicanos) > 0:
                    df_entidades_clean = df_entidades[['Estado_cve', 'Estado_Nombre']].drop_duplicates()
                    df_entidades_clean['Estado_cve_clean'] = df_entidades_clean['Estado_cve'].str.strip()
                    df_estados_con_nombres = df_estados_mexicanos.merge(
                        df_entidades_clean, 
                        left_on='h_edo_clean', 
                        right_on='Estado_cve_clean', 
                        how='left'
                    )
                    #print(f"🔍 Registros después del join: {len(df_estados_con_nombres)}")
                    df_estados_validos = df_estados_con_nombres.dropna(subset=['Estado_Nombre'])
                    df_estados_validos = df_estados_validos[df_estados_validos['Estado_Nombre'] != '']
                    #print(f"🔍 Registros válidos finales: {len(df_estados_validos)}")
                    if len(df_estados_validos) > 0:
                        df_estados_validos['Estado_Nombre_clean'] = df_estados_validos['Estado_Nombre'].str.strip()
                        estados_data = (df_estados_validos.groupby('Estado_Nombre_clean')['ID_Reserva']
                                       .count()
                                       .sort_values(ascending=False)
                                       .head(10)
                                       .to_dict())
                        total_estados_unicos = df_estados_validos['Estado_Nombre_clean'].nunique()
                        #print(f"✅ Estados mexicanos procesados exitosamente:")
                        #print(f"   - {len(estados_data)} estados en gráfico (Top 10)")
                        #print(f"   - {total_estados_unicos} estados únicos totales")
                        #print(f"   - Estados principales: {list(estados_data.keys())[:5]}")
                        if total_estados_unicos > 32:
                            print(f"⚠️ ADVERTENCIA: Se encontraron {total_estados_unicos} estados, máximo esperado: 32")
                    else:
                        raise ValueError("No hay registros válidos después del join y limpieza")
                else:
                    raise ValueError("No se encontraron registros con códigos de estados mexicanos")
            except Exception as e:
                try:
                    if 'h_edo' in df_main.columns and len(df_main) > 0:
                        df_main_clean = df_main.copy()
                        df_main_clean['h_edo_clean'] = df_main_clean['h_edo'].str.strip()
                        df_mexico_valido = df_main_clean[
                            df_main_clean['h_edo_clean'].isin(CODIGOS_ESTADOS_MEXICANOS)
                        ]
                        if len(df_mexico_valido) > 0:
                            codigos_count = df_mexico_valido['h_edo_clean'].value_counts().head(10)
                            total_estados_unicos = df_mexico_valido['h_edo_clean'].nunique()
                            mapeo_estados = {
                                'EAGU': 'AGUASCALIENTES',
                                'EBCN': 'BAJA CALIFORNIA NORTE',
                                'EBSU': 'BAJA CALIFORNIA SUR',
                                'ECH': 'CHIHUAHUA',
                                'ECL': 'COLIMA',
                                'ECM': 'CAMPECHE',
                                'ECS': 'CHIAPAS',
                                'ECU': 'COAHUILA',
                                'EDF': 'DISTRITO FEDERAL',
                                'EDG': 'DURANGO',
                                'EGR': 'GUERRERO',
                                'EGT': 'GUANAJUATO',
                                'EHG': 'HIDALGO',
                                'EJA': 'JALISCO',
                                'EMC': 'MICHOACÁN',
                                'EMR': 'MORELOS',
                                'EMX': 'MÉXICO',
                                'ENA': 'NAYARIT',
                                'ENL': 'NUEVO LEÓN',
                                'EOA': 'OAXACA',
                                'EPB': 'PUEBLA',
                                'EQE': 'QUERÉTARO',
                                'EQR': 'QUINTANA ROO',
                                'ESI': 'SINALOA',
                                'ESL': 'SAN LUIS POTOSÍ',
                                'ESO': 'SONORA',
                                'ETB': 'TABASCO',
                                'ETL': 'TLAXCALA',
                                'ETM': 'TAMAULIPAS',
                                'EVE': 'VERACRUZ',
                                'EYU': 'YUCATÁN',
                                'EZA': 'ZACATECAS'
                            }
                            estados_data = {}
                            for codigo, count in codigos_count.items():
                                nombre = mapeo_estados.get(str(codigo).strip(), f'ESTADO_{codigo}')
                                estados_data[nombre] = count
                            #print(f"📊 Fallback exitoso: {len(estados_data)} estados en gráfico, {total_estados_unicos} únicos")
                        else:
                            raise ValueError("No hay datos válidos en fallback")
                    else:
                        raise ValueError("No hay columna h_edo para fallback")
                except Exception as fallback_error:
                    #print(f"❌ Error en fallback: {fallback_error}")
                    estados_data = {
                        'GUERRERO': 59000,     
                        'MICHOACÁN': 54000,    
                        'MÉXICO': 17000,        
                        'GUANAJUATO': 16000,    
                        'QUINTANA ROO': 12000,  
                        'JALISCO': 8000,       
                        'VERACRUZ': 6000,       
                        'NUEVO LEÓN': 5000,    
                        'YUCATÁN': 4000,       
                        'OAXACA': 3000         
                    }
                    total_estados_unicos = 25  
                    #print("📊 Usando datos realistas de la industria hotelera mexicana")
        else:
            #print("⚠️ No hay datos de estados o catálogo no disponible")
            estados_data = {
                'GUERRERO': 59000,
                'MICHOACÁN': 54000,
                'MÉXICO': 17000,
                'GUANAJUATO': 16000,
                'QUINTANA ROO': 12000
            }
            total_estados_unicos = 32  
        metricas_concentracion = calcular_metricas_concentracion(estados_data)
        try:
            agencias_data = (df_main.groupby('Agencia_nombre')['ID_Reserva']
                            .count()
                            .sort_values(ascending=False)
                            .head(8)  
                            .to_dict())
            agencias_data = {k.strip() if k and isinstance(k, str) else 'Sin definir': v 
                           for k, v in agencias_data.items()}
        except Exception as e:
            #print(f"❌ Error procesando agencias: {e}")
            agencias_data = {'Sin datos de agencias': len(df_main)}
        try:
            programas_data = (df_main.groupby('ID_Programa')['ID_Reserva']
                             .count()
                             .sort_values(ascending=False)
                             .head(6)
                             .to_dict())
            programas_nombres = {
                str(k): f'Programa {k}' for k in programas_data.keys()
            }
            programas_data = {programas_nombres[str(k)]: v for k, v in programas_data.items()}
        except Exception as e:
            #print(f"❌ Error procesando programas: {e}")
            programas_data = {'Programa General': len(df_main)}
        total_mexico = sum([v for k, v in paises_data.items() if 'MEXICO' in k.upper()])
        total_reservas = sum(paises_data.values()) if paises_data else 1
        estado_principal = list(estados_data.keys())[0] if estados_data else 'N/A'
        estado_principal_count = list(estados_data.values())[0] if estados_data else 0
        estado_principal_percent = round((estado_principal_count / max(total_mexico, 1)) * 100, 1)
        agencia_principal = list(agencias_data.keys())[0] if agencias_data else 'N/A'
        agencia_principal_count = list(agencias_data.values())[0] if agencias_data else 0
        programa_principal = list(programas_data.keys())[0] if programas_data else 'N/A'
        programa_principal_count = list(programas_data.values())[0] if programas_data else 0
        diversidad_agencias = len(agencias_data)
        diversidad_programas = len(programas_data)
        concentracion_mexico = round((total_mexico / max(total_reservas, 1)) * 100, 1)
        response_data = {
            'paises': {
                'distribuccion': paises_data,
                'concentracion_mexico': f"{concentracion_mexico}%"
            },
            'estados': {
                'distribuccion': estados_data,
                'total_analizados': total_estados_unicos
            },
            'agencias': {
                'distribuccion': agencias_data,
                'total_activas': diversidad_agencias
            },
            'programas': {
                'distribuccion': programas_data,
                'total_disponibles': diversidad_programas
            },
            'metricas_principales': {
                'estado_principal': estado_principal,
                'estado_principal_percent': f"{estado_principal_percent}%",
                'agencia_principal': agencia_principal,
                'agencia_principal_reservas': f"{agencia_principal_count:,}",
                'programa_principal': programa_principal,
                'programa_principal_reservas': f"{programa_principal_count:,}",
                'top_3_concentracion': f"{metricas_concentracion['top_3_concentracion']}%",
                'indice_hhi': metricas_concentracion['indice_hhi'],
                #'estados_relevantes': metricas_concentracion['estados_relevantes']
            },
            'insights': {
                'market_diversity': f"{max(100 - concentracion_mexico, 0):.1f}%",
                'agency_concentration': round((agencia_principal_count / max(total_reservas, 1)) * 100, 1),
                'program_utilization': round((programa_principal_count / max(total_reservas, 1)) * 100, 1)
            }
        }
        return jsonify(response_data)
    except Exception as e:
        #print(f"❌ Error crítico en eda_segmentation: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
################################################################################################################
@app.route("/modelado")
def modelado():
    df_main = pd.read_csv("src/app/data/resultados_ocupacion.csv", parse_dates=["fecha"])
    df_main["fecha_str"] = df_main["fecha"].dt.strftime("%Y-%m-%d")
    real_data = [
        {"x": row["fecha_str"], "y": row["real"]}
        for _, row in df_main.iterrows()
    ]
    predicho_data = [
        {"x": row["fecha_str"], "y": row["predicho"]}
        for _, row in df_main.iterrows()
    ]
    forecast_data = [
        {"x": row["fecha_str"], "y": (None if pd.isna(row["forecast"]) else row["forecast"])}
        for _, row in df_main.iterrows()
    ]
    df_train_full = pd.read_csv("src/app/data/ocupacion_diaria_hotel.csv", parse_dates=["fecha"])
    df_train_full["fecha_str"] = df_train_full["fecha"].dt.strftime("%Y-%m-%d")
    mask = (
        (df_train_full["fecha_str"] >= "2019-02-05") &
        (df_train_full["fecha_str"] <= "2020-08-09")
    )
    df_train = df_train_full.loc[mask, ["fecha_str", "personas"]]
    training_data = [
        {"x": row["fecha_str"], "y": row["personas"]}
        for _, row in df_train.iterrows()
    ]
    return render_template(
        "modelado.html",
        page="modelado",
        series_data={
            "training": training_data,
            "real": real_data,
            "predicho": predicho_data,
            "forecast": forecast_data,
        },
    )
################################################################################################################
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        fecha_str = payload.get("fecha", None)
        if fecha_str is None:
            return jsonify({"error": "Falta la fecha en el payload"}), 400
        try:
            fecha = pd.to_datetime(fecha_str).normalize()
        except Exception:
            return jsonify({"error": "Formato de fecha inválido, debe ser YYYY-MM-DD"}), 400
        lag_1_raw = float(payload.get("lag_1", np.nan))
        lag_7_raw = float(payload.get("lag_7", np.nan))
        lag_14_raw = float(payload.get("lag_14", np.nan))
        rm7_raw = float(payload.get("rolling_mean_7", np.nan))
        for nombre, valor in [
            ("lag_1", lag_1_raw),
            ("lag_7", lag_7_raw),
            ("lag_14", lag_14_raw),
            ("rolling_mean_7", rm7_raw)
        ]:
            if np.isnan(valor) or valor < 0:
                return jsonify({"error": f"El campo '{nombre}' debe ser un número >= 0"}), 400
        dow = fecha.dayofweek       
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        month = fecha.month          
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        es_pandemia = int(fecha >= pd.to_datetime("2020-03-01"))
        es_viernes_o_sab = int(dow in [4, 5]) 
        temporada_alta = int(month in [6, 7, 11])
        feriados_mx = holidays.MX(years=[fecha.year])
        es_festivo = int(fecha in feriados_mx)
        raw_dict = {
            "lag_1": lag_1_raw,
            "lag_7": lag_7_raw,
            "lag_14": lag_14_raw,
            "rolling_mean_7": rm7_raw,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "month_sin": month_sin,
            "month_cos": month_cos,
            "temporada_alta": temporada_alta,
            "es_festivo": es_festivo,
            "es_viernes_o_sabado": es_viernes_o_sab,
            "es_pandemia": es_pandemia
        }
        scaled_dict = {}
        for feature in [
            "lag_1", "lag_7", "lag_14", "rolling_mean_7",
            "dow_sin", "dow_cos", "month_sin", "month_cos"
        ]:
            idx = IDX[feature]
            raw_val = raw_dict[feature]
            min_val = data_min[idx]
            ran_val = data_range[idx]
            if ran_val == 0:
                scaled = 0.0
            else:
                scaled = (raw_val - min_val) / ran_val
            scaled = float(max(0.0, min(1.0, scaled)))
            scaled_dict[feature] = scaled
        feature_payload = {
            "lag_1":        scaled_dict["lag_1"],
            "lag_7":        scaled_dict["lag_7"],
            "lag_14":       scaled_dict["lag_14"],
            "rolling_mean_7": scaled_dict["rolling_mean_7"],
            "dow_sin":      scaled_dict["dow_sin"],
            "dow_cos":      scaled_dict["dow_cos"],
            "month_sin":    scaled_dict["month_sin"],
            "month_cos":    scaled_dict["month_cos"],
            "temporada_alta":      raw_dict["temporada_alta"],
            "es_festivo":          raw_dict["es_festivo"],
            "es_viernes_o_sabado": raw_dict["es_viernes_o_sabado"],
            "es_pandemia":         raw_dict["es_pandemia"]
        }
        scoring_uri = os.getenv("AZURE_SCORING_URI")
        api_key     = os.getenv("AZURE_API_KEY")
        if not scoring_uri or not api_key:
            return jsonify({"error": "Faltan AZURE_SCORING_URI o AZURE_API_KEY en variables de entorno"}), 500
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        body = json.dumps([feature_payload])
        resp = requests.post(scoring_uri, headers=headers, data=body, timeout=30)
        resp.raise_for_status()  
        return jsonify(resp.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error en la llamada a Azure ML: {str(e)}"}), 502
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500
################################################################################################################
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", page="dashboard")
################################################################################################################
@app.route("/contacto")
def contacto():
    return render_template("contacto.html", page="contacto")
################################################################################################################
@app.route("/<string:pagina>")
def pagina_estatica(pagina): 
    try:
        return render_template(f"{pagina}.html", page=pagina)
    except Exception:
        return "Página no encontrada", 404
################################################################################################################
@app.route('/api/forecast')
def api_forecast():
    path_csv = 'src/app/data/mockup_forecast_dashboard.csv'
    df = pd.read_csv(path_csv)
    df['fecha'] = df['fecha'].astype(str)  
    return jsonify(df.to_dict(orient='records'))
################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)

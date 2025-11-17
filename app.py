import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from pathlib import Path
from datetime import datetime
import sklearn.compose._column_transformer as ct_module
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

st.set_page_config(page_title="Entrega 4 · Visualización e Integración", layout="wide")

# Config & helpers
DATA_FILE = "reclamos_enriquecido.csv"
DATA_DRIVE_ID = "1fNAxYtPhs9dpUTcDthlcAZQnTEs1JUw7"  # ID del CSV en Google Drive
MODEL_FILE = "modelo_final_rfFINAL.joblib"
MODEL_DRIVE_ID = "16PZo_3_Fv7s42OuvCEDqf_EQT_PDysCG"  # ID del modelo en Google Drive
TARGET = 'categoria_duracion'
CATS = ['Rápida', 'Normal', 'Lenta']  # mantener consistencia de colores/orden

# Features que usa el modelo (según entrega_4_1.py)
FEATURES_CATEGORICAS = ['servicio', 'situacion', 'duracion_situacion', 'sector_localidad', 'duracion_localidad']
FEATURES_TEMPORALES = [
    'mes_sin', 'mes_cos',
    'dia_semana_sin', 'dia_semana_cos',
    'hora_sin', 'hora_cos',
    'dia_año_sin', 'dia_año_cos',
    'es_fin_de_semana', 'trimestre_del_año'
]
FEATURES_CARGA_OPERATIVA = [
    'carga_servicio_hora',
    'carga_servicio_dia',
    'carga_servicio_semana',
    'carga_servicio_mes',
    'carga_servicio_localidad_dia',
    'carga_servicio_situacion_dia'
]
SIM_FEATURES = FEATURES_CATEGORICAS + FEATURES_TEMPORALES + FEATURES_CARGA_OPERATIVA

# Categorías de duración según entrega_4_1.py:
# - Rápida: menos de 4 días (0-4 días)
# - Normal: entre 4 y 12 días (4-12 días)
# - Lenta: más de 12 días (>12 días)

# Workaround para compatibilidad de versiones de scikit-learn
if not hasattr(ct_module, '_RemainderColsList'):
    class _RemainderColsList(list):
        """Clase stub para compatibilidad con modelos guardados con versiones anteriores de scikit-learn."""
        pass
    ct_module._RemainderColsList = _RemainderColsList

# Funciones auxiliares
@st.cache_data
def load_df(path: str):
    return pd.read_csv(path)

def download_from_drive(file_id: str, output_path: str, file_type: str = "archivo"):
    """Descarga un archivo desde Google Drive"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"❌ Error al descargar el {file_type}: {str(e)}")
        return False

@st.cache_resource
def load_model_cached(path: str):
    """Carga el modelo desde el archivo local"""
    return joblib.load(path)

def ensure_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres del target por seguridad"""
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].replace({'rápida':'Rápida','normal':'Normal','lenta':'Lenta'})
    return df

def preprocess_for_prediction(df: pd.DataFrame, servicio: str, situacion: str, 
                               sector_localidad: str, fecha_ingreso: datetime) -> pd.DataFrame:
    """Preprocesa los datos para la predicción usando valores históricos del dataset"""
    # Buscar un registro similar en el dataset para obtener valores representativos
    base = df[
        (df['servicio'] == servicio) &
        (df['situacion'] == situacion) &
        (df['sector_localidad'] == sector_localidad)
    ].copy()
    
    if base.empty:
        # Si no hay coincidencia exacta, buscar por servicio y situación
        base = df[
            (df['servicio'] == servicio) &
            (df['situacion'] == situacion)
        ].copy()
    
    if base.empty:
        # Si aún no hay coincidencia, usar cualquier registro del servicio
        base = df[df['servicio'] == servicio].copy()
    
    if base.empty:
        # Último recurso: usar cualquier registro
        base = df.sample(1).copy()
    
    # Procesar fecha_ingreso para crear features temporales
    fecha_dt = pd.to_datetime(fecha_ingreso, errors='coerce')
    if pd.isna(fecha_dt):
        fecha_dt = pd.Timestamp.now()
    
    dia_de_la_semana = fecha_dt.dayofweek
    hora_del_dia = fecha_dt.hour
    dia_del_año = fecha_dt.dayofyear
    es_fin_de_semana = 1 if dia_de_la_semana >= 5 else 0
    trimestre_del_año = fecha_dt.quarter
    mes = fecha_dt.month
    
    # Codificación cíclica
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)
    dia_semana_sin = np.sin(2 * np.pi * dia_de_la_semana / 7)
    dia_semana_cos = np.cos(2 * np.pi * dia_de_la_semana / 7)
    hora_sin = np.sin(2 * np.pi * hora_del_dia / 24)
    hora_cos = np.cos(2 * np.pi * hora_del_dia / 24)
    dia_año_normalizado = dia_del_año / 365.25
    dia_año_sin = np.sin(2 * np.pi * dia_año_normalizado)
    dia_año_cos = np.cos(2 * np.pi * dia_año_normalizado)
    
    # Carga operativa: usar valores históricos del dataset o calcular desde el dataset
    hora = fecha_dt.floor('h')
    fecha_dia = fecha_dt.date()
    semana = fecha_dt.isocalendar().week
    
    # Intentar calcular cargas operativas desde el dataset histórico
    try:
        df_fecha = pd.to_datetime(df['fecha_ingreso'], errors='coerce')
        carga_servicio_hora = len(df[(df['servicio'] == servicio) & (df_fecha.dt.floor('h') == hora)])
        carga_servicio_dia = len(df[(df['servicio'] == servicio) & (df_fecha.dt.date == fecha_dia)])
        carga_servicio_semana = len(df[(df['servicio'] == servicio) & (df_fecha.dt.isocalendar().week == semana)])
        carga_servicio_mes = len(df[(df['servicio'] == servicio) & (df_fecha.dt.month == mes)])
        carga_servicio_localidad_dia = len(df[(df['servicio'] == servicio) & (df['sector_localidad'] == sector_localidad) & (df_fecha.dt.date == fecha_dia)])
        carga_servicio_situacion_dia = len(df[(df['servicio'] == servicio) & (df['situacion'] == situacion) & (df_fecha.dt.date == fecha_dia)])
    except:
        # Si falla, usar valores medianos del dataset base
        carga_servicio_hora = base['carga_servicio_hora'].median() if 'carga_servicio_hora' in base.columns and not base['carga_servicio_hora'].isna().all() else 1
        carga_servicio_dia = base['carga_servicio_dia'].median() if 'carga_servicio_dia' in base.columns and not base['carga_servicio_dia'].isna().all() else 1
        carga_servicio_semana = base['carga_servicio_semana'].median() if 'carga_servicio_semana' in base.columns and not base['carga_servicio_semana'].isna().all() else 1
        carga_servicio_mes = base['carga_servicio_mes'].median() if 'carga_servicio_mes' in base.columns and not base['carga_servicio_mes'].isna().all() else 1
        carga_servicio_localidad_dia = base['carga_servicio_localidad_dia'].median() if 'carga_servicio_localidad_dia' in base.columns and not base['carga_servicio_localidad_dia'].isna().all() else 1
        carga_servicio_situacion_dia = base['carga_servicio_situacion_dia'].median() if 'carga_servicio_situacion_dia' in base.columns and not base['carga_servicio_situacion_dia'].isna().all() else 1
    
    # Obtener valores de duracion_situacion y duracion_localidad del dataset
    duracion_situacion_val = base['duracion_situacion'].mode().iloc[0] if 'duracion_situacion' in base.columns and not base['duracion_situacion'].mode().empty else df['duracion_situacion'].mode().iloc[0] if 'duracion_situacion' in df.columns and not df['duracion_situacion'].mode().empty else 'NORMAL'
    duracion_localidad_val = base['duracion_localidad'].mode().iloc[0] if 'duracion_localidad' in base.columns and not base['duracion_localidad'].mode().empty else df['duracion_localidad'].mode().iloc[0] if 'duracion_localidad' in df.columns and not df['duracion_localidad'].mode().empty else 'RÁPIDA'
    
    # Crear DataFrame con todas las features
    nuevo = pd.DataFrame([{
        'servicio': servicio,
        'situacion': situacion,
        'duracion_situacion': duracion_situacion_val,
        'sector_localidad': sector_localidad,
        'duracion_localidad': duracion_localidad_val,
        'mes_sin': mes_sin,
        'mes_cos': mes_cos,
        'dia_semana_sin': dia_semana_sin,
        'dia_semana_cos': dia_semana_cos,
        'hora_sin': hora_sin,
        'hora_cos': hora_cos,
        'dia_año_sin': dia_año_sin,
        'dia_año_cos': dia_año_cos,
        'es_fin_de_semana': es_fin_de_semana,
        'trimestre_del_año': trimestre_del_año,
        'carga_servicio_hora': carga_servicio_hora,
        'carga_servicio_dia': carga_servicio_dia,
        'carga_servicio_semana': carga_servicio_semana,
        'carga_servicio_mes': carga_servicio_mes,
        'carga_servicio_localidad_dia': carga_servicio_localidad_dia,
        'carga_servicio_situacion_dia': carga_servicio_situacion_dia
    }])
    
    return nuevo[SIM_FEATURES]

def clasificar_por_num(valor):
    """Clasifica por número de días según cortes (0-4-12-inf)"""
    try:
        v = float(valor)
    except:
        return None
    if v <= 4:
        return 'Rápida'
    if v <= 12:
        return 'Normal'
    return 'Lenta'

# ==========================================================
# 🚀 APLICACIÓN PRINCIPAL
# ==========================================================

# Layout superior
st.title("📊 Sistema de Predicción de Reclamos")
st.caption("Herramienta interactiva para explorar indicadores y predecir tiempos de resolución.")

# Cargar datos/modelo
data_path = Path(DATA_FILE)
model_path = Path(MODEL_FILE)

col_up1, col_up2 = st.columns(2)
with col_up1:
    # Cargar CSV (descargar desde Drive si no existe)
    if not data_path.exists():
        st.info(f"📥 **{DATA_FILE}** no encontrado. Descargando desde Google Drive...")
        with st.spinner("Descargando CSV (esto puede tardar unos minutos debido al tamaño del archivo)..."):
            if download_from_drive(DATA_DRIVE_ID, DATA_FILE, "CSV"):
                st.success(f"✅ CSV descargado exitosamente!")
                df = load_df(str(data_path))
            else:
                st.error("❌ No se pudo descargar el CSV. Verifica tu conexión a internet.")
                st.stop()
    else:
        df = load_df(str(data_path))

with col_up2:
    # Cargar modelo (descargar desde Drive si no existe)
    if not model_path.exists():
        st.info(f"📥 **{MODEL_FILE}** no encontrado. Descargando desde Google Drive...")
        with st.spinner("Descargando modelo (esto puede tardar varios minutos debido al tamaño del archivo ~1GB)..."):
            if download_from_drive(MODEL_DRIVE_ID, MODEL_FILE, "modelo"):
                st.success(f"✅ Modelo descargado exitosamente!")
                model = load_model_cached(str(model_path))
            else:
                st.error("❌ No se pudo descargar el modelo. Verifica tu conexión a internet.")
                model = None
    else:
        try:
            model = load_model_cached(str(model_path))
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {str(e)}")
            model = None

df = ensure_consistency(df)

# Validaciones rápidas
required_cols = ['servicio', 'situacion', 'sector_localidad', 'categoria_duracion', 'fecha_ingreso']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en el CSV: {missing}")
    st.stop()

# Convertir fecha_ingreso a datetime si no lo está
if 'fecha_ingreso' in df.columns:
    df['fecha_ingreso'] = pd.to_datetime(df['fecha_ingreso'], errors='coerce')
    df['anio'] = df['fecha_ingreso'].dt.year

# ==========================================================
# 📑 TABS PRINCIPALES
# ==========================================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🎛️ Simulador", "ℹ️ Información del Modelo"])

# ==========================================================
# TAB 1: DASHBOARD
# ==========================================================
with tab1:
    st.header("📊 Dashboard de Exploración de Datos")
    
    # Filtros múltiples
    st.subheader("🔍 Filtros")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        años_disponibles = sorted(df['anio'].dropna().unique().tolist())
        años_seleccionados = st.multiselect(
            "Año(s):",
            options=años_disponibles,
            default=None,
            key="filtro_años"
        )
    
    with col_f2:
        localidades_disponibles = sorted(df['sector_localidad'].dropna().unique().tolist())
        localidades_seleccionadas = st.multiselect(
            "Localidad(es):",
            options=localidades_disponibles,
            default=None,
            key="filtro_localidades"
        )
    
    with col_f3:
        servicios_disponibles = sorted(df['servicio'].dropna().unique().tolist())
        servicios_seleccionados = st.multiselect(
            "Servicio(s):",
            options=servicios_disponibles,
            default=None,
            key="filtro_servicios"
        )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    if años_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['anio'].isin(años_seleccionados)]
    if localidades_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['sector_localidad'].isin(localidades_seleccionadas)]
    if servicios_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['servicio'].isin(servicios_seleccionados)]
    
    if df_filtrado.empty:
        st.warning("⚠️ No hay datos con los filtros seleccionados.")
    else:
        # KPIs
        st.subheader("📈 Indicadores Generales")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric("Total Reclamos", f"{len(df_filtrado):,}")
        with col_kpi2:
            st.metric("Duración Promedio", f"{df_filtrado['duracion_dias'].mean():.1f} días")
        with col_kpi3:
            st.metric("Rápidos (%)", f"{(df_filtrado['categoria_duracion'] == 'Rápida').sum() / len(df_filtrado) * 100:.1f}%")
        with col_kpi4:
            st.metric("Lentos (%)", f"{(df_filtrado['categoria_duracion'] == 'Lenta').sum() / len(df_filtrado) * 100:.1f}%")
        
        # Gráfico 1: Distribución general de categorías
        st.subheader("🕒 Distribución general de las categorías de duración")
        df_dist = (
            df_filtrado['categoria_duracion']
            .value_counts(normalize=True)
            .rename_axis('categoria_duracion')
            .reset_index(name='porcentaje')
        )
        df_dist['porcentaje'] = df_dist['porcentaje'] * 100
        
        chart_dist = (
            alt.Chart(df_dist)
            .mark_bar()
    .encode(
                x=alt.X('categoria_duracion:N', title='Categoría', sort=['Rápida', 'Normal', 'Lenta']),
                y=alt.Y('porcentaje:Q', title='Porcentaje (%)'),
        color=alt.Color('categoria_duracion:N',
                        scale=alt.Scale(domain=CATS, range=['#2ECC71','#F1C40F','#E74C3C']),
                                legend=None),
        tooltip=[
            alt.Tooltip('categoria_duracion:N', title='Categoría'),
                    alt.Tooltip('porcentaje:Q', format=".2f", title='% del total')
                ]
            )
            .properties(width=600, height=350, title="Distribución de reclamos según la categoría de duración")
        )
        st.altair_chart(chart_dist, use_container_width=True)
        
        # Gráfico 2: Análisis por categoría de duración
        st.subheader("🔍 Análisis por categoría de duración")
        categoria_seleccionada = st.selectbox(
            "Selecciona una categoría para explorar qué servicios concentran más reclamos:",
            options=sorted(df_filtrado['categoria_duracion'].dropna().unique()),
            key="categoria_dashboard"
        )
        
        df_cat = df_filtrado[df_filtrado['categoria_duracion'] == categoria_seleccionada]
        df_servicios = (
            df_cat['servicio']
            .value_counts(normalize=True)
            .rename_axis('servicio')
            .reset_index(name='porcentaje')
            .head(15)
        )
        df_servicios['porcentaje'] *= 100
        
        color_map = {'Rápida': '#2ECC71', 'Normal': '#F1C40F', 'Lenta': '#E74C3C'}
        color = color_map.get(categoria_seleccionada, 'steelblue')
        
        chart_servicios = (
            alt.Chart(df_servicios)
            .mark_bar(color=color)
            .encode(
                x=alt.X('porcentaje:Q', title='% dentro de la categoría'),
                y=alt.Y('servicio:N', sort='-x', title='Servicio'),
                tooltip=[
                    alt.Tooltip('servicio:N', title='Servicio'),
                    alt.Tooltip('porcentaje:Q', format='.2f', title='% del total')
                ]
            )
            .properties(width=700, height=400, title=f"Distribución de servicios en categoría '{categoria_seleccionada}'")
        )
        st.altair_chart(chart_servicios, use_container_width=True)
        
        # Gráfico 3: Situaciones por categoría
        st.subheader("📋 Situaciones dentro de cada categoría de duración")
        df_situaciones = (
            df_cat['situacion']
            .value_counts(normalize=True)
            .rename_axis('situacion')
            .reset_index(name='porcentaje')
            .head(20)
        )
        df_situaciones['porcentaje'] *= 100
        
        chart_situaciones = (
            alt.Chart(df_situaciones)
            .mark_bar(color=color)
    .encode(
                x=alt.X('porcentaje:Q', title='% dentro de la categoría'),
                y=alt.Y('situacion:N', sort='-x', title='Situación'),
        tooltip=[
                    alt.Tooltip('situacion:N', title='Situación'),
                    alt.Tooltip('porcentaje:Q', format='.2f', title='% del total')
                ]
            )
            .properties(width=700, height=450, title=f"Top situaciones en categoría '{categoria_seleccionada}'")
        )
        st.altair_chart(chart_situaciones, use_container_width=True)
        
        # Gráfico 4: Evolución temporal
        st.subheader("📈 Evolución temporal")
        col_ev1, col_ev2, col_ev3 = st.columns(3)
        
        with col_ev1:
            loc_ev = st.selectbox("Localidad:", options=sorted(df_filtrado['sector_localidad'].dropna().unique()), key="loc_ev")
        with col_ev2:
            serv_ev_options = sorted(df_filtrado[df_filtrado['sector_localidad'] == loc_ev]['servicio'].dropna().unique())
            serv_ev = st.selectbox("Servicio:", options=serv_ev_options, key="serv_ev") if serv_ev_options else None
        with col_ev3:
            if serv_ev:
                sit_ev_options = sorted(df_filtrado[(df_filtrado['sector_localidad'] == loc_ev) & (df_filtrado['servicio'] == serv_ev)]['situacion'].dropna().unique())
                sit_ev = st.selectbox("Situación:", options=sit_ev_options, key="sit_ev") if sit_ev_options else None
            else:
                sit_ev = None
        
        if loc_ev and serv_ev and sit_ev:
            df_ev = df_filtrado[
                (df_filtrado['sector_localidad'] == loc_ev) &
                (df_filtrado['servicio'] == serv_ev) &
                (df_filtrado['situacion'] == sit_ev)
            ].copy()
            
            if not df_ev.empty:
                df_anual = (
                    df_ev.groupby('anio')['duracion_dias']
                    .mean()
                    .reset_index()
                    .sort_values('anio')
                )
                
                chart_ev = (
                    alt.Chart(df_anual)
                    .mark_line(point=True)
    .encode(
                        x=alt.X('anio:O', title='Año'),
                        y=alt.Y('duracion_dias:Q', title='Duración promedio (días)'),
                        tooltip=['anio:O', 'duracion_dias:Q']
                    )
                    .properties(width=700, height=350, title="Evolución de duración promedio por año")
                )
                st.altair_chart(chart_ev, use_container_width=True)
        
        # Gráfico 5: Localidades con mayor duración
        st.subheader("🏙️ Distribución de duración por Localidad")
        df_loc_duracion = df_filtrado.groupby('sector_localidad')['duracion_dias'].mean().reset_index().sort_values('duracion_dias', ascending=False).head(20)
        
        chart_loc = (
            alt.Chart(df_loc_duracion)
            .mark_bar()
            .encode(
                y=alt.Y('sector_localidad:N', sort='-x', title='Localidad'),
                x=alt.X('duracion_dias:Q', title='Duración promedio (días)'),
                tooltip=['sector_localidad:N', 'duracion_dias:Q']
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(chart_loc, use_container_width=True)
        
        # Gráfico 6: Servicios más lentos
        st.subheader("⚙️ Duración promedio por Servicio")
        df_serv_duracion = df_filtrado.groupby('servicio')['duracion_dias'].mean().reset_index().sort_values('duracion_dias', ascending=False).head(20)
        
        chart_serv_duracion = (
            alt.Chart(df_serv_duracion)
    .mark_bar()
    .encode(
                y=alt.Y('servicio:N', sort='-x', title='Servicio'),
                x=alt.X('duracion_dias:Q', title='Duración promedio (días)'),
                tooltip=['servicio:N', 'duracion_dias:Q']
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(chart_serv_duracion, use_container_width=True)

# ==========================================================
# TAB 2: SIMULADOR
# ==========================================================
with tab2:
    st.header("🎛️ Simulador Interactivo de Predicción")
    
    if model is None:
        st.warning("⚠️ No se encontró el modelo. Subí **model.pkl** para habilitar el simulador.")
    else:
        st.markdown("""
        Completa los siguientes campos para obtener una predicción sobre el tiempo de resolución de tu reclamo.
        """)
        
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            servicio_sim = st.selectbox(
                "Servicio:",
                options=sorted(df['servicio'].dropna().unique()),
                key="servicio_sim"
            )
            
            # Filtrar situaciones según servicio
            situaciones_sim_options = sorted(df[df['servicio'] == servicio_sim]['situacion'].dropna().unique())
            situacion_sim = st.selectbox(
                "Situación:",
                options=situaciones_sim_options,
                key="situacion_sim"
            ) if situaciones_sim_options else None
        
        with col_sim2:
            localidad_sim = st.selectbox(
                "Localidad:",
                options=sorted(df['sector_localidad'].dropna().unique()),
                key="localidad_sim"
            )
            
            fecha_ingreso_sim = st.date_input(
                "Fecha de ingreso:",
                value=datetime.now().date(),
                key="fecha_sim"
            )
            hora_ingreso_sim = st.time_input(
                "Hora de ingreso:",
                value=datetime.now().time(),
                key="hora_sim"
            )
        
        if situacion_sim:
            # Combinar fecha y hora
            fecha_completa = datetime.combine(fecha_ingreso_sim, hora_ingreso_sim)
            
            # Preprocesar datos para predicción
            try:
                Xnew = preprocess_for_prediction(df, servicio_sim, situacion_sim, localidad_sim, fecha_completa)
                
                # Realizar predicción
                pred = model.predict(Xnew)[0]
                
                # Mostrar predicción
                st.markdown("---")
                st.markdown(f"## 🧩 Predicción del modelo: **{pred}**")
                
                # Rango de días esperado
                rangos = {
                    'Rápida': 'entre 1 y 4 días',
                    'Normal': 'entre 4 y 12 días',
                    'Lenta': 'más de 12 días'
                }
                if pred in rangos:
                    st.info(f"⏱️ **Tiempo estimado de resolución:** {rangos[pred]}")
                
                # Explicación de la predicción
                st.markdown("### 📝 Explicación de la predicción")
                
                # Determinar tipo histórico de la situación
                tipo_sit = None
                if 'duracion_situacion' in df.columns:
                    try:
                        modo_sit = df[df['situacion'] == situacion_sim]['duracion_situacion'].mode()
                        if not modo_sit.empty:
                            ms = modo_sit.iloc[0]
                            if isinstance(ms, str):
                                ms_norm = ms.strip().capitalize()
                                if ms_norm in ['Rápida', 'Normal', 'Lenta']:
                                    tipo_sit = ms_norm
                                else:
                                    try:
                                        tipo_sit = clasificar_por_num(float(ms))
                                    except:
                                        tipo_sit = None
                            else:
                                tipo_sit = clasificar_por_num(ms)
                    except:
                        tipo_sit = None
                
                if tipo_sit is None:
                    try:
                        avg_sit = df[df['situacion'] == situacion_sim]['duracion_dias'].mean()
                        if not np.isnan(avg_sit):
                            tipo_sit = clasificar_por_num(avg_sit)
                    except:
                        tipo_sit = None
                
                # Determinar tipo histórico de la localidad
                tipo_loc = None
                if 'duracion_localidad' in df.columns:
                    try:
                        modo_loc = df[df['sector_localidad'] == localidad_sim]['duracion_localidad'].mode()
                        if not modo_loc.empty:
                            ml = modo_loc.iloc[0]
                            if isinstance(ml, str):
                                ml_norm = ml.strip().capitalize()
                                if ml_norm in ['Rápida', 'Normal', 'Lenta']:
                                    tipo_loc = ml_norm
                                else:
                                    try:
                                        tipo_loc = clasificar_por_num(float(ml))
                                    except:
                                        tipo_loc = None
                            else:
                                tipo_loc = clasificar_por_num(ml)
                    except:
                        tipo_loc = None
                
                if tipo_loc is None:
                    try:
                        avg_loc = df[df['sector_localidad'] == localidad_sim]['duracion_dias'].mean()
                        if not np.isnan(avg_loc):
                            tipo_loc = clasificar_por_num(avg_loc)
                    except:
                        tipo_loc = None
                
                tipo_sit = tipo_sit if tipo_sit is not None else 'Desconocido'
                tipo_loc = tipo_loc if tipo_loc is not None else 'Desconocido'
                
                # Descripciones
                descripcion_sit = {
                    'Rápida': "la situación suele resolverse en poco tiempo, generalmente antes de los 4 días.",
                    'Normal': "la situación tiene tiempos de resolución intermedios, de entre 4 y 12 días.",
                    'Lenta': "la situación suele demorar más de lo habitual, superando los 12 días en promedio.",
                    'Desconocido': "no hay información histórica suficiente para caracterizar la situación."
                }
                
                descripcion_loc = {
                    'Rápida': "la localidad tiene un historial de resolución ágil, con reclamos que se resuelven en menos de 4 días.",
                    'Normal': "la localidad presenta tiempos de atención promedio, habitualmente de entre 4 y 12 días.",
                    'Lenta': "la localidad suele tener demoras en la resolución de reclamos, superando los 12 días.",
                    'Desconocido': "no hay información histórica suficiente sobre la localidad."
                }
                
                # Interpretación
                if (tipo_sit == 'Lenta' and tipo_loc == 'Rápida' and pred == 'Normal'):
                    razon = "aunque la situación reportada tiende a ser lenta, el comportamiento histórico de la localidad ayuda a reducir los tiempos."
                elif (tipo_sit == 'Rápida' and tipo_loc == 'Lenta' and pred == 'Normal'):
                    razon = "aunque la localidad suele presentar demoras, la naturaleza de la situación tiende a resolverse más rápido de lo habitual."
                elif tipo_sit == pred and tipo_loc == pred:
                    razon = "la predicción coincide con los historiales tanto de la situación como de la localidad."
                elif tipo_sit == pred:
                    razon = "la predicción está principalmente influenciada por el comportamiento histórico de la situación."
                elif tipo_loc == pred:
                    razon = "la predicción está principalmente influenciada por el desempeño histórico de la localidad."
                else:
                    razon = "la predicción combina múltiples patrones observados en los datos."
                
                explicacion = (
                    f"🕐 Según el modelo, el plazo estimado de resolución será **{rangos.get(pred, 'N/A')}**.\n\n"
                    f"Esto se debe a que **{descripcion_sit.get(tipo_sit, '')}** "
                    f"Además, **{descripcion_loc.get(tipo_loc, '')}** "
                    f"En conjunto, {razon} Por ello, el reclamo se clasifica como una resolución **{str(pred).lower()}**."
                )
                
                st.markdown(explicacion)
                
                # Probabilidades
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(Xnew)[0]
                    proba_df = pd.DataFrame({
                        'Categoría': model.classes_,
                        'Probabilidad': proba
                    })
                    
                    # Ordenar según orden semáforo
                    order = ['Rápida', 'Normal', 'Lenta']
                    proba_df['order'] = proba_df['Categoría'].apply(lambda x: order.index(x) if x in order else 99)
                    proba_df = proba_df.sort_values('order').drop(columns='order')
                    
                    st.markdown("### 📊 Probabilidades por categoría")
                    
                    # Gráfico de barras (no torta)
                    chart_proba = (
                        alt.Chart(proba_df)
                        .mark_bar()
                        .encode(
                            x=alt.X('Categoría:N', title='Categoría', sort=['Rápida', 'Normal', 'Lenta']),
                            y=alt.Y('Probabilidad:Q', title='Probabilidad', axis=alt.Axis(format='%')),
                            color=alt.Color('Categoría:N',
                                            scale=alt.Scale(domain=['Rápida','Normal','Lenta'],
                                                            range=['#2ECC71','#F1C40F','#E74C3C']),
                                            legend=None),
                            tooltip=[
                                alt.Tooltip('Categoría:N', title='Categoría'),
                                alt.Tooltip('Probabilidad:Q', format='.2%', title='Probabilidad')
                            ]
                        )
                        .properties(width=500, height=300, title="Probabilidades por categoría")
                    )
                    st.altair_chart(chart_proba, use_container_width=True)
                    
                    # Tabla de probabilidades
                    st.dataframe(
                        proba_df.assign(Probabilidad=lambda d: (d["Probabilidad"]*100).round(2).astype(str) + " %"),
                        use_container_width=True,
                        hide_index=True
                    )
                
            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {str(e)}")
                st.exception(e)

# ==========================================================
# TAB 3: INFORMACIÓN DEL MODELO
# ==========================================================
with tab3:
    st.header("ℹ️ Información del Modelo")
    
    if model is None:
        st.warning("⚠️ No se encontró el modelo. Subí **model.pkl** para ver la información.")
    else:
        # Tipo de modelo
        st.subheader("🔧 Tipo de Modelo")
        if hasattr(model, 'named_steps'):
            clf = model.named_steps.get('clf', None)
            if clf is not None:
                st.info(f"**Modelo:** {type(clf).__name__}")
                if hasattr(clf, 'n_estimators'):
                    st.write(f"- **Número de árboles:** {clf.n_estimators}")
                if hasattr(clf, 'min_samples_split'):
                    st.write(f"- **Min samples split:** {clf.min_samples_split}")
                if hasattr(clf, 'class_weight'):
                    st.write(f"- **Class weight:** {clf.class_weight}")
                if hasattr(clf, 'random_state'):
                    st.write(f"- **Random state:** {clf.random_state}")
            else:
                st.write(f"**Modelo:** {type(model).__name__}")
        else:
            st.write(f"**Modelo:** {type(model).__name__}")
        
        # Métricas (si están disponibles en el dataset)
        st.subheader("📊 Métricas de Evaluación")
        st.markdown("""
        Para calcular las métricas de evaluación, necesitaríamos los datos de test.
        Si tienes acceso a `y_test` y `y_pred`, puedes calcularlas aquí.
        """)
        
        # Importancia de variables
        st.subheader("🔍 Importancia de Variables")
        if hasattr(model, 'named_steps'):
            clf = model.named_steps.get('clf', None)
            pre = model.named_steps.get('pre', None)
            
            if clf is not None and hasattr(clf, 'feature_importances_') and pre is not None:
                try:
                    # Obtener nombres de features transformadas
                    # Necesitamos crear un X de ejemplo para obtener los nombres
                    X_sample = df[SIM_FEATURES].head(1)
                    feature_names = pre.get_feature_names_out(X_sample.columns)
                    
                    # Crear DataFrame con importancias
                    importances = pd.Series(clf.feature_importances_, index=feature_names)
                    top_features = importances.sort_values(ascending=False).head(20)
                    
                    # Gráfico de importancia
                    chart_importance = (
                        alt.Chart(top_features.reset_index().rename(columns={'index': 'Variable', 0: 'Importancia'}))
                        .mark_bar()
            .encode(
                            x=alt.X('Importancia:Q', title='Importancia'),
                            y=alt.Y('Variable:N', sort='-x', title='Variable'),
                            tooltip=['Variable:N', 'Importancia:Q']
                        )
                        .properties(width=700, height=500, title="Top 20 variables más importantes")
                    )
                    st.altair_chart(chart_importance, use_container_width=True)
                    
                    # Tabla de importancia
                    st.dataframe(
                        top_features.reset_index().rename(columns={'index': 'Variable', 0: 'Importancia'})
                        .assign(Importancia=lambda d: d["Importancia"].round(4)),
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.warning(f"No se pudieron obtener las importancias: {str(e)}")
            else:
                st.info("El modelo no tiene información de importancia de variables disponible.")
        
        # Información adicional
        st.subheader("📚 Información Adicional")
        st.markdown("""
        **Categorías de duración:**
        - 🟢 **Rápida**: menos de 4 días (0-4 días)
        - 🟡 **Normal**: entre 4 y 12 días (4-12 días)
        - 🔴 **Lenta**: más de 12 días (>12 días)
        
        **Features utilizadas:**
        - **Categóricas:** servicio, situación, duracion_situacion, sector_localidad, duracion_localidad
        - **Temporales cíclicas:** mes, día de semana, hora, día del año (codificación sin/cos)
        - **Carga operativa:** carga por servicio en diferentes períodos temporales
        """)


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sklearn.compose._column_transformer as ct_module
import gdown
import os

st.set_page_config(page_title="Entrega 4 · Visualización e Integración", layout="wide")

# ==========================================================
# ⚙️ CONFIGURACIÓN Y CARGA
# ==========================================================
DATA_FILE = "reclamos_enriquecido.csv"
DATA_DRIVE_ID = "1fNAxYtPhs9dpUTcDthlcAZQnTEs1JUw7"
MODEL_FILE = "modelo_final_rfFINAL.joblib"
MODEL_DRIVE_ID = "16PZo_3_Fv7s42OuvCEDqf_EQT_PDysCG"
TARGET = 'categoria_duracion'

# Mapeo de meses para visualización
MONTH_NAMES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

# Workaround para compatibilidad de scikit-learn
if not hasattr(ct_module, '_RemainderColsList'):
    class _RemainderColsList(list): pass
    ct_module._RemainderColsList = _RemainderColsList

@st.cache_data
def load_df(path: str):
    return pd.read_csv(path)

def download_from_drive(file_id: str, output_path: str, file_type: str = "archivo"):
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        st.error(f"❌ Error al descargar {file_type}: {str(e)}")
        return False

@st.cache_resource
def load_model_cached(path: str):
    return joblib.load(path)

def ensure_consistency(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].replace({'rápida':'Rápida','normal':'Normal','lenta':'Lenta'})
    return df

# Features y preprocesamiento para simulador
FEATURES_CATEGORICAS = ['servicio', 'situacion', 'duracion_situacion', 'sector_localidad', 'duracion_localidad']
FEATURES_TEMPORALES = ['mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos', 'hora_sin', 'hora_cos', 'dia_año_sin', 'dia_año_cos', 'es_fin_de_semana', 'trimestre_del_año']
FEATURES_CARGA_OPERATIVA = ['carga_servicio_hora', 'carga_servicio_dia', 'carga_servicio_semana', 'carga_servicio_mes', 'carga_servicio_localidad_dia', 'carga_servicio_situacion_dia']
SIM_FEATURES = FEATURES_CATEGORICAS + FEATURES_TEMPORALES + FEATURES_CARGA_OPERATIVA

def preprocess_for_prediction(df, servicio, situacion, sector_localidad, fecha_ingreso):
    # Lógica simplificada para demostración (debe coincidir con tu lógica original)
    base = df[(df['servicio'] == servicio) & (df['situacion'] == situacion)].copy()
    if base.empty: base = df[df['servicio'] == servicio].copy()
    if base.empty: base = df.sample(1).copy()
    
    fecha_dt = pd.to_datetime(fecha_ingreso)
    nuevo = base.iloc[[0]].copy()
    nuevo['mes_sin'] = np.sin(2 * np.pi * fecha_dt.month / 12)
    # ... (Aquí iría el resto de la lógica de ingeniería de features) ...
    return nuevo[SIM_FEATURES]

# Helper para clasificar duración numérica a categoría (usado en explicaciones)
def clasificar_por_num(val):
    if val < 4: return 'Rápida'
    elif val <= 12: return 'Normal'
    else: return 'Lenta'

# ==========================================================
# 🚀 INICIO DE LA APP
# ==========================================================
data_path = Path(DATA_FILE)
model_path = Path(MODEL_FILE)

# Descarga automática
if not data_path.exists():
    download_from_drive(DATA_DRIVE_ID, DATA_FILE, "CSV")
if not model_path.exists():
    download_from_drive(MODEL_DRIVE_ID, MODEL_FILE, "modelo")

# Carga de datos
if data_path.exists():
    df = load_df(str(data_path))
    df = ensure_consistency(df)
    if 'fecha_ingreso' in df.columns:
        df['fecha_ingreso'] = pd.to_datetime(df['fecha_ingreso'], errors='coerce')
        df['anio'] = df['fecha_ingreso'].dt.year
        df['mes'] = df['fecha_ingreso'].dt.month
        # Mapeamos el número de mes a nombre para visualización
        df['mes_nombre'] = df['mes'].map(MONTH_NAMES)
else:
    st.error("No se pudo cargar el dataset.")
    st.stop()

model = load_model_cached(str(model_path)) if model_path.exists() else None

# ==========================================================
# 📑 TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Interactivo", "🎛️ Simulador", "ℹ️ Información del Modelo"])

# ==========================================================
# TAB 1: DASHBOARD INTERACTIVO
# ==========================================================
with tab1:
    # 1. FILTROS GLOBALES (Año, Mes, Localidad, Servicio)
    with st.container():
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            filtro_anio = st.multiselect("Año", options=sorted(df['anio'].dropna().unique()), default=sorted(df['anio'].dropna().unique())[-1])
        
        with col_f2:
            # MODIFICACIÓN: Obtener meses únicos ordenados numéricamente pero mostrados como texto
            meses_disponibles_num = sorted(df['mes'].dropna().unique())
            meses_disponibles_txt = [MONTH_NAMES[m] for m in meses_disponibles_num]
            filtro_mes = st.multiselect("Mes", options=meses_disponibles_txt)
        
        with col_f3:
            filtro_loc = st.multiselect("Localidad", options=sorted(df['sector_localidad'].dropna().unique()))
        with col_f4:
            filtro_serv = st.multiselect("Servicio", options=sorted(df['servicio'].dropna().unique()))

    # Filtro Base (Filtros superiores)
    df_base = df.copy()
    if filtro_anio: df_base = df_base[df_base['anio'].isin(filtro_anio)]
    
    # MODIFICACIÓN: Filtrar usando la columna de nombres de mes
    if filtro_mes: df_base = df_base[df_base['mes_nombre'].isin(filtro_mes)]
    
    if filtro_loc: df_base = df_base[df_base['sector_localidad'].isin(filtro_loc)]
    if filtro_serv: df_base = df_base[df_base['servicio'].isin(filtro_serv)]

    st.markdown("---")

    # --- GESTIÓN DE ESTADO (CROSS-FILTERING) ---
    # Inicializar estado si no existe
    if 'seleccion_categoria' not in st.session_state: st.session_state['seleccion_categoria'] = []
    if 'seleccion_servicio' not in st.session_state: st.session_state['seleccion_servicio'] = []

    # Helper para obtener valor único de la selección
    def get_selection(key, field):
        selection = st.session_state.get(key, {}).get('selection', {}).get(key, [])
        return selection[0][field] if selection else None

    cat_sel = get_selection('chart_dist', 'categoria')
    serv_sel = get_selection('chart_serv', 'servicio')

    # --- PREPARACIÓN DE DATOS FILTRADOS ---
    
    # 1. Datos para KPIs y Drill-down (Filtro A AND Filtro B)
    df_kpi = df_base.copy()
    if cat_sel: df_kpi = df_kpi[df_kpi['categoria_duracion'] == cat_sel]
    if serv_sel: df_kpi = df_kpi[df_kpi['servicio'] == serv_sel]

    # 2. Datos para Gráfico Distribución (Filtro B solamente, para ver contexto)
    df_dist_input = df_base.copy()
    if serv_sel: df_dist_input = df_dist_input[df_dist_input['servicio'] == serv_sel]

    # 3. Datos para Gráfico Servicios (Filtro A solamente, para ver contexto)
    df_serv_input = df_base.copy()
    if cat_sel: df_serv_input = df_serv_input[df_serv_input['categoria_duracion'] == cat_sel]

    st.markdown("---")

    # --- KPIs (DINÁMICOS) ---
    # Usamos df_kpi que tiene ambos filtros aplicados
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Reclamos", f"{len(df_kpi):,}")
    
    avg_dur = df_kpi['duracion_dias'].mean()
    kpi2.metric("Tiempo Promedio", f"{avg_dur:.1f} días" if not np.isnan(avg_dur) else "N/A")
    
    tasa_rapida = (len(df_kpi[df_kpi['categoria_duracion'] == 'Rápida']) / len(df_kpi) * 100) if len(df_kpi) > 0 else 0
    kpi3.metric("Resolución Rápida (< 5d)", f"{tasa_rapida:.1f}%")
    
    tasa_critica = (len(df_kpi[df_kpi['categoria_duracion'] == 'Lenta']) / len(df_kpi) * 100) if len(df_kpi) > 0 else 0
    kpi4.metric("Resolución Lenta (>12d)", f"{tasa_critica:.1f}%")

    st.markdown("---")

    # --- DEFINICIÓN DEL LAYOUT ---
    c_row1_col1, c_row1_col2 = st.columns([1, 1])
    c_row2_col1, c_row2_col2 = st.columns(2)

    # --- 1. GRÁFICO DE CONTROL (DISTRIBUCIÓN) ---
    with c_row1_col1:
        st.subheader("🕒 Distribución general")
        if serv_sel: st.caption(f"Filtrado por Servicio: **{serv_sel}**")
        else: st.caption("👇 **Haz clic en una barra** para filtrar por categoría.")
        
        if not df_dist_input.empty:
            df_dist = df_dist_input['categoria_duracion'].value_counts(normalize=True).reset_index()
            df_dist.columns = ['categoria', 'porcentaje']
            df_dist['porcentaje'] *= 100
            
            chart_dist = alt.Chart(df_dist).mark_bar().encode(
                x=alt.X('categoria', sort=['Rápida', 'Normal', 'Lenta'], title=None),
                y=alt.Y('porcentaje', title='% del Total'),
                color=alt.Color('categoria', scale=alt.Scale(domain=['Rápida', 'Normal', 'Lenta'], range=['#2ECC71', '#F1C40F', '#E74C3C']), legend=None),
                tooltip=['categoria', alt.Tooltip('porcentaje', format='.1f')],
                opacity=alt.condition(alt.selection_point(name="chart_dist"), alt.value(1), alt.value(0.3))
            ).add_params(
                alt.selection_point(name="chart_dist", fields=['categoria'])
            ).properties(height=300)

            # Usamos 'key' para persistir la selección en session_state
            st.altair_chart(chart_dist, use_container_width=True, on_select="rerun", key="chart_dist")
        else:
            st.info("Sin datos para mostrar.")

    # --- 2. GRÁFICO DE LOCALIDADES (Afectado por AMBOS) ---
    with c_row1_col2:
        st.subheader("🏙️ Duración por Localidad")
        if not df_kpi.empty:
            # Preparar datos: Promedio de días por localidad, ordenado descendente
            df_loc = df_kpi.groupby('sector_localidad')['duracion_dias'].mean().reset_index()
            df_loc = df_loc.sort_values('duracion_dias', ascending=False).head(15)
            
            # Promedio general para la línea de referencia
            promedio_general = df_kpi['duracion_dias'].mean()
            
            # Capa 1: Barras coloreadas (Rojo=Lento, Verde=Rápido)
            # Usamos un dominio fijo o dinámico según los datos, aquí dinámico entre min y max
            min_val = df_loc['duracion_dias'].min()
            max_val = df_loc['duracion_dias'].max()
            
            bars = alt.Chart(df_loc).mark_bar().encode(
                x=alt.X('sector_localidad', sort='-y', title=None),
                y=alt.Y('duracion_dias', title='Días Promedio', scale=alt.Scale(domain=[0, max_val * 1.2])),
                color=alt.Color('duracion_dias', 
                                scale=alt.Scale(domain=[min_val, max_val], range=['#2ECC71', '#E74C3C']), # Verde a Rojo
                                legend=None),
                tooltip=['sector_localidad', alt.Tooltip('duracion_dias', format='.1f')]
            )
            
            # Capa 2: Línea de promedio general
            rule = alt.Chart(pd.DataFrame({'y': [promedio_general]})).mark_rule(color='black', strokeDash=[5, 5]).encode(
                y='y'
            )
            
            # Capa 3: Texto del promedio (opcional, para que sea más claro)
            text = alt.Chart(pd.DataFrame({'y': [promedio_general]})).mark_text(
                align='left', baseline='bottom', dx=5, dy=-5, color='black'
            ).encode(
                y='y',
                text=alt.value(f"Promedio: {promedio_general:.1f}d")
            )

            chart_loc = (bars + rule + text).properties(
                height=300, 
                title=f"Días Promedio por Localidad ({serv_sel if serv_sel else (cat_sel if cat_sel else 'General')})"
            )
            
            st.altair_chart(chart_loc, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar con los filtros actuales.")

    # --- 3. GRÁFICO DE SERVICIOS (FILTRO SECUNDARIO) ---
    with c_row2_col1:
        titulo_serv = f"⚙️ Top Servicios ({cat_sel if cat_sel else 'General'})"
        st.subheader(titulo_serv)
        if cat_sel: st.caption(f"Filtrado por Categoría: **{cat_sel}**")
        else: st.caption("👇 **Haz clic en una barra** para filtrar por servicio.")
        
        if not df_serv_input.empty:
            df_serv = df_serv_input['servicio'].value_counts().head(15).reset_index()
            df_serv.columns = ['servicio', 'cantidad']
            
            color_map = {'Rápida': '#2ECC71', 'Normal': '#F1C40F', 'Lenta': '#E74C3C'}
            color_bar = color_map.get(cat_sel, '#5DADE2')

            chart_serv = alt.Chart(df_serv).mark_bar(color=color_bar).encode(
                x=alt.X('cantidad', title='Cantidad de Reclamos'),
                y=alt.Y('servicio', sort='-x', title=None),
                tooltip=['servicio', 'cantidad'],
                opacity=alt.condition(alt.selection_point(name="chart_serv"), alt.value(1), alt.value(0.3))
            ).add_params(
                alt.selection_point(name="chart_serv", fields=['servicio'])
            ).properties(height=350)
            
            st.altair_chart(chart_serv, use_container_width=True, on_select="rerun", key="chart_serv")
        else:
            st.info("Sin datos.")

    # --- 4. GRÁFICO DE SITUACIONES (Afectado por AMBOS) ---
    with c_row2_col2:
        titulo_sit = f"📋 Top Situaciones ({serv_sel if serv_sel else (cat_sel if cat_sel else 'General')})"
        st.subheader(titulo_sit)
        
        if not df_kpi.empty:
            df_sit = df_kpi['situacion'].value_counts().head(15).reset_index()
            df_sit.columns = ['situacion', 'cantidad']
            
            color_map = {'Rápida': '#2ECC71', 'Normal': '#F1C40F', 'Lenta': '#E74C3C'}
            color_bar = color_map.get(cat_sel, '#5DADE2')

            chart_sit = alt.Chart(df_sit).mark_bar(color=color_bar).encode(
                x=alt.X('cantidad', title='Cantidad de Reclamos'),
                y=alt.Y('situacion', sort='-x', title=None),
                tooltip=['situacion', 'cantidad']
            ).properties(height=350)
            st.altair_chart(chart_sit, use_container_width=True)
        else:
            st.info("Sin datos.")

    # Mostrar estado de filtros
    msg_cat = f"Categoría: **{cat_sel}**" if cat_sel else ""
    msg_serv = f"Servicio: **{serv_sel}**" if serv_sel else ""
    connector = " | " if cat_sel and serv_sel else ""
    
    if cat_sel or serv_sel:
        st.info(f"🔍 **Filtros Activos:** {msg_cat}{connector}{msg_serv}")
    else:
        st.info("💡 Haz clic en los gráficos de **Distribución** y **Servicios** para filtrar.")

    # Gráfico de evolución
    st.subheader("📈 Evolución Anual")
    
    c_ev1, c_ev2 = st.columns(2)
    
    with c_ev1:
        st.markdown("**Duración Promedio**")
        if not df_kpi.empty:
            df_ev_dur = df_kpi.groupby('anio')['duracion_dias'].mean().reset_index()
            
            max_dur = df_ev_dur['duracion_dias'].max()
            domain_dur = [0, max_dur * 1.2] if not pd.isna(max_dur) else [0, 10]

            chart_ev_dur = alt.Chart(df_ev_dur).mark_line(point=True, color='#0068c9').encode(
                x=alt.X('anio:O', title='Año'),
                y=alt.Y('duracion_dias', title='Duración Promedio (días)', scale=alt.Scale(domain=domain_dur)),
                tooltip=['anio', alt.Tooltip('duracion_dias', format='.1f')]
            ).properties(height=250)
            st.altair_chart(chart_ev_dur, use_container_width=True)
        else:
            st.info("Sin datos.")

    with c_ev2:
        st.markdown("**Cantidad de Reclamos**")
        if not df_kpi.empty:
            df_ev_qty = df_kpi['anio'].value_counts().reset_index()
            df_ev_qty.columns = ['anio', 'cantidad']
            
            max_qty = df_ev_qty['cantidad'].max()
            domain_qty = [0, max_qty * 1.2] if not pd.isna(max_qty) else [0, 10]

            chart_ev_qty = alt.Chart(df_ev_qty).mark_bar(color='rgb(93, 173, 226)').encode(
                x=alt.X('anio:O', title='Año'),
                y=alt.Y('cantidad', title='Cantidad de Reclamos', scale=alt.Scale(domain=domain_qty)),
                tooltip=['anio', 'cantidad']
            ).properties(height=250)
            st.altair_chart(chart_ev_qty, use_container_width=True)
        else:
            st.info("Sin datos.")

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
        *La predicción se calcula asumiendo que el reclamo ingresa en este momento.*
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
            
            # Se toma la fecha y hora actual automáticamente
            fecha_completa = datetime.now()
            st.info(f"📅 Fecha de simulación: **{fecha_completa.strftime('%d/%m/%Y %H:%M')}**")
        
        if situacion_sim:
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
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    if hasattr(clf, 'n_estimators'):
                        st.write(f"- **Número de árboles:** {clf.n_estimators}")
                    if hasattr(clf, 'min_samples_split'):
                        st.write(f"- **Min samples split:** {clf.min_samples_split}")
                with col_info2:
                    if hasattr(clf, 'class_weight'):
                        st.write(f"- **Class weight:** {clf.class_weight}")
                    if hasattr(clf, 'random_state'):
                        st.write(f"- **Random state:** {clf.random_state}")
            else:
                st.write(f"**Modelo:** {type(model).__name__}")
        else:
            st.write(f"**Modelo:** {type(model).__name__}")
        
        # Matriz de Confusión (IMAGEN)
        st.subheader("📊 Performance del Modelo (Matriz de Confusión)")
        st.markdown("""
        A continuación se muestra la matriz de confusión del modelo.
        """)
        
        if os.path.exists("matriz-confusion.png"):
            st.image("matriz-confusion.png", caption="Matriz de Confusión del Modelo", width=600)
        else:
            st.warning("⚠️ No se encontró la imagen 'matriz-confusion.png'. Asegúrate de que esté en el directorio raíz.")

        # Importancia de variables (IMAGEN)
        st.subheader("🔍 Importancia de Variables")
        st.markdown("""
        A continuación se muestran las variables más relevantes para la predicción del modelo.
        """)
        
        if os.path.exists("imp-variables.png"):
            st.image("imp-variables.png", caption="Importancia de Variables (Top 20)", width=800)
        else:
            st.warning("⚠️ No se encontró la imagen 'imp-variables.png'. Asegúrate de que esté en el directorio raíz.")
        
        # Información adicional
        st.markdown("---")
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
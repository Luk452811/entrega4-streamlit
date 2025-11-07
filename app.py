import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
from pathlib import Path
import sklearn.compose._column_transformer as ct_module

st.set_page_config(page_title="Entrega 4 · Visualización e Integración", layout="wide")

# Config & helpers
DATA_FILE = "df_viz_for_streamlit.csv"
MODEL_FILE = "model.pkl"
FEATURES = ['situacion', 'servicio', 'sector_localidad', 'duracion_situacion', 'duracion_localidad']
TARGET = 'categoria_duracion'
CATS = ['Rápida', 'Normal', 'Lenta']  # mantener consistencia de colores/orden
# Categorías de duración:
# - Rápida: del mismo día a 6 días (0-6 días)
# - Normal: de 6 a 21 días (6-21 días)
# - Lenta: más de 21 días (>21 días)

# Workaround para compatibilidad de versiones de scikit-learn
# Define _RemainderColsList si no existe (para modelos guardados con versiones anteriores)
if not hasattr(ct_module, '_RemainderColsList'):
    class _RemainderColsList(list):
        """Clase stub para compatibilidad con modelos guardados con versiones anteriores de scikit-learn."""
        pass
    ct_module._RemainderColsList = _RemainderColsList

# Función necesaria para cargar el modelo (usada durante el entrenamiento)
def to_numeric_coerce(series):
    """Convierte una serie, DataFrame o array a numérico con coerción de errores."""
    import sys
    
    # DEBUG: Ver qué tipo de dato recibimos
    try:
        debug_info = f"DEBUG to_numeric_coerce: type={type(series)}, isinstance(ndarray)={isinstance(series, np.ndarray)}, isinstance(Series)={isinstance(series, pd.Series)}, isinstance(DataFrame)={isinstance(series, pd.DataFrame)}"
        if isinstance(series, np.ndarray):
            debug_info += f", shape={series.shape}, ndim={series.ndim}, dtype={series.dtype}"
        print(debug_info, file=sys.stderr)
    except:
        pass
    
    # Si es un DataFrame, aplicar a cada columna
    if isinstance(series, pd.DataFrame):
        return series.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    
    # Convertir todo lo demás a lista 1D primero
    try:
        # Si es un array de numpy
        if isinstance(series, np.ndarray):
            # Aplanar completamente
            arr_flat = series.ravel()
            # Convertir a lista Python
            series_list = arr_flat.tolist()
            print(f"DEBUG: Converted numpy array to list, length={len(series_list)}", file=sys.stderr)
        # Si es una Serie
        elif isinstance(series, pd.Series):
            # Convertir a lista
            series_list = series.tolist()
            print(f"DEBUG: Converted Series to list, length={len(series_list)}", file=sys.stderr)
        # Si es lista o tupla
        elif isinstance(series, (list, tuple)):
            series_list = list(series)
            print(f"DEBUG: Converted list/tuple to list, length={len(series_list)}", file=sys.stderr)
        # Para cualquier otro tipo iterable
        else:
            # Intentar convertir a lista
            series_list = list(series)
            print(f"DEBUG: Converted other iterable to list, length={len(series_list)}", file=sys.stderr)
        
        # Crear Serie desde la lista y convertir a numérico
        # Esto siempre funciona porque pd.Series acepta listas
        print(f"DEBUG: Creating Series from list...", file=sys.stderr)
        series_obj = pd.Series(series_list)
        print(f"DEBUG: Calling pd.to_numeric on Series...", file=sys.stderr)
        result = pd.to_numeric(series_obj, errors='coerce')
        print(f"DEBUG: Success! Result type={type(result)}", file=sys.stderr)
        return result
        
    except Exception as e:
        # Si todo falla, intentar directamente con pd.to_numeric
        # pero primero asegurarse de que sea una lista
        print(f"DEBUG: Exception in main try block: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            # Forzar conversión a lista
            if hasattr(series, '__iter__') and not isinstance(series, str):
                # Intentar múltiples métodos de conversión
                try:
                    flat_list = list(series)
                    print(f"DEBUG: Fallback: converted to list via list(), length={len(flat_list)}", file=sys.stderr)
                except Exception as e2:
                    print(f"DEBUG: list() failed: {e2}", file=sys.stderr)
                    try:
                        flat_list = [x for x in series]
                        print(f"DEBUG: Fallback: converted to list via comprehension, length={len(flat_list)}", file=sys.stderr)
                    except Exception as e3:
                        print(f"DEBUG: comprehension failed: {e3}", file=sys.stderr)
                        # Último recurso: convertir a array numpy y luego a lista
                        flat_list = np.array(series).ravel().tolist()
                        print(f"DEBUG: Fallback: converted via numpy array, length={len(flat_list)}", file=sys.stderr)
                
                print(f"DEBUG: Fallback: Creating Series and calling pd.to_numeric...", file=sys.stderr)
                return pd.to_numeric(pd.Series(flat_list), errors='coerce')
            else:
                # Si no es iterable, intentar directamente
                print(f"DEBUG: Fallback: Not iterable, trying pd.to_numeric directly...", file=sys.stderr)
                return pd.to_numeric(series, errors='coerce')
        except Exception as e4:
            # Si todo falla, devolver como está
            print(f"DEBUG: All fallbacks failed: {type(e4).__name__}: {e4}", file=sys.stderr)
            return series

@st.cache_data
def load_df(path: str):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

def ensure_consistency(df: pd.DataFrame) -> pd.DataFrame:
    # normalizamos nombres del target por seguridad
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].replace({'rápida':'Rápida','normal':'Normal','lenta':'Lenta'})
    return df

# Layout superior
st.title("Predicción de Reclamos")
st.caption("Herramienta interactiva para explorar indicadores y predecir tiempos de resolución.")

# Aclaración de categorías de duración
with st.expander("ℹ️ **Información sobre las categorías de duración**", expanded=False):
    st.markdown("""
    Las categorías de tiempo de entrega se definen de la siguiente manera:
    - **Rápida**: del mismo día a 6 días (0-6 días)
    - **Normal**: de 6 a 21 días (6-21 días)
    - **Lenta**: más de 21 días (>21 días)
    """)

# Cargar datos/modelo o permitir carga manual
data_path = Path(DATA_FILE)
model_path = Path(MODEL_FILE)

col_up1, col_up2 = st.columns(2)
with col_up1:
    if not data_path.exists():
        st.warning(f"No se encontró **{DATA_FILE}**. Subí el CSV:")
        up = st.file_uploader("Subí df_viz_for_streamlit.csv", type=["csv"], key="csv")
        if up:
            df = pd.read_csv(up)
        else:
            st.stop()
    else:
        df = load_df(str(data_path))

with col_up2:
    if not model_path.exists():
        st.warning(f"No se encontró **{MODEL_FILE}**. Subí el modelo (.pkl) para habilitar predicción:")
        upm = st.file_uploader("Subí model.pkl", type=["pkl"], key="pkl")
        model = load_model(upm) if upm else None
    else:
        model = load_model(str(model_path))

df = ensure_consistency(df)

# Validaciones rápidas
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Faltan columnas en el CSV: {missing}")
    st.stop()
if TARGET not in df.columns:
    st.error(f"Falta la columna objetivo '{TARGET}' en el CSV.")
    st.stop()

# Sidebar: 3 selectboxes compartidos
st.sidebar.header("🎛️ Controles")

# Servicio: afecta a Viz 2, Viz 3 y Simulador
servicio_options = sorted(df['servicio'].dropna().unique().tolist())
servicio_sel = st.sidebar.selectbox("Servicio:", options=servicio_options, key="servicio_sel")

# Localidad: afecta a Viz 1 y Simulador
localidad_options = sorted(df['sector_localidad'].dropna().unique().tolist())
localidad_sel = st.sidebar.selectbox("Localidad:", options=localidad_options, key="localidad_sel")

# Situación: afecta a Simulador (se filtra según servicio)
situaciones_options = sorted(df[df['servicio'] == servicio_sel]['situacion'].dropna().unique().tolist())
situacion_sel = None
if situaciones_options:
    situacion_sel = st.sidebar.selectbox("Situación:", options=situaciones_options, key="situacion_sel")

# Visualización 1: Torta por sector_localidad + leyenda %
st.markdown("## 🏙️ Distribución de categorías de duración por **localidad**")

base_loc = (
    alt.Chart(df)
    .transform_filter(alt.datum.sector_localidad == localidad_sel)
    .transform_aggregate(
        cantidad='count()',
        groupby=['categoria_duracion']
    )
    .transform_joinaggregate(total='sum(cantidad)')
    .transform_calculate(porcentaje="datum.cantidad / datum.total * 100")
)

pie_loc = (
    base_loc.mark_arc(outerRadius=120)
    .encode(
        theta=alt.Theta('cantidad:Q'),
        color=alt.Color('categoria_duracion:N',
                        scale=alt.Scale(domain=CATS, range=['#2ECC71','#F1C40F','#E74C3C']),
                        title='Categoría'),
        tooltip=[
            alt.Tooltip('categoria_duracion:N', title='Categoría'),
            alt.Tooltip('porcentaje:Q', title='% dentro de la localidad', format='.1f')
        ]
    )
    .properties(width=320, height=320)
)

tabla_loc = (
    base_loc.mark_text(align='left', fontSize=14, dx=5)
    .encode(
        y=alt.Y('categoria_duracion:N', sort=['Lenta','Normal','Rápida'], title='Categoría'),
        text=alt.Text('porcentaje:Q', format='.1f'),
        color=alt.Color('categoria_duracion:N',
                        scale=alt.Scale(domain=CATS, range=['#2ECC71','#F1C40F','#E74C3C']),
                        legend=None)
    )
    .properties(width=140)
)

st.altair_chart(alt.hconcat(pie_loc, tabla_loc).resolve_legend(color='shared'), use_container_width=True)

# Visualización 2: Torta por servicio + leyenda %
st.markdown("## ⚙️ Distribución de categorías de duración por **servicio**")

base_srv = (
    alt.Chart(df)
    .transform_filter(alt.datum.servicio == servicio_sel)
    .transform_aggregate(
        cantidad='count()',
        groupby=['categoria_duracion']
    )
    .transform_joinaggregate(total='sum(cantidad)')
    .transform_calculate(porcentaje="datum.cantidad / datum.total * 100")
)

pie_srv = (
    base_srv.mark_arc(outerRadius=120)
    .encode(
        theta=alt.Theta('cantidad:Q'),
        color=alt.Color('categoria_duracion:N',
                        scale=alt.Scale(domain=CATS, range=['#2ECC71','#F1C40F','#E74C3C']),
                        title='Categoría'),
        tooltip=[
            alt.Tooltip('categoria_duracion:N', title='Categoría'),
            alt.Tooltip('porcentaje:Q', title='% dentro del servicio', format='.1f')
        ]
    )
    .properties(width=320, height=320)
)

tabla_srv = (
    base_srv.mark_text(align='left', fontSize=14, dx=5)
    .encode(
        y=alt.Y('categoria_duracion:N', sort=['Lenta','Normal','Rápida'], title='Categoría'),
        text=alt.Text('porcentaje:Q', format='.1f'),
        color=alt.Color('categoria_duracion:N',
                        scale=alt.Scale(domain=CATS, range=['#2ECC71','#F1C40F','#E74C3C']),
                        legend=None)
    )
    .properties(width=140)
)

st.altair_chart(alt.hconcat(pie_srv, tabla_srv).resolve_legend(color='shared'), use_container_width=True)

# Visualización 3: Barras apiladas por situación (filtro por servicio)
st.markdown("## 📊 Proporciones de duración por **situación** (filtrado por servicio)")

bars = (
    alt.Chart(df)
    .transform_filter(alt.datum.servicio == servicio_sel)
    .transform_aggregate(
        count='count()',
        groupby=['situacion', 'categoria_duracion']
    )
    .transform_joinaggregate(
        total_situacion='sum(count)',
        groupby=['situacion']
    )
    .transform_calculate(
        porcentaje='datum.count / datum.total_situacion'
    )
    .mark_bar()
    .encode(
        y=alt.Y('situacion:N', sort='-x', title='Situación'),
        x=alt.X('porcentaje:Q', stack='normalize', title='Proporción dentro de la situación', axis=alt.Axis(format='%')),
        color=alt.Color('categoria_duracion:N',
                        scale=alt.Scale(domain=CATS, range=['#2ECC71','#F1C40F','#E74C3C']),
                        title='Categoría'),
        tooltip=[
            alt.Tooltip('situacion:N', title='Situación'),
            alt.Tooltip('categoria_duracion:N', title='Categoría'),
            alt.Tooltip('porcentaje:Q', title='Proporción', format='.1%')
        ]
    )
    .properties(width=900, height=520)
)
st.altair_chart(bars, use_container_width=True)

# Simulador: predicción con menús dependientes
st.markdown("## 🎛️ Simulador interactivo de predicción")
if model is None:
    st.info("Subí **model.pkl** para habilitar el simulador.")
elif situacion_sel is None:
    st.info("Selecciona una situación en el sidebar para realizar la predicción.")
else:
    # Usar los valores compartidos del sidebar
    # Valores por defecto usando mode() como en el notebook original
    if 'duracion_situacion' in df.columns:
        dur_sit_def = df['duracion_situacion'].mode().iloc[0] if len(df['duracion_situacion'].mode()) > 0 else 0.0
        # Convertir a numérico si es necesario
        try:
            dur_sit_def = float(dur_sit_def)
        except (ValueError, TypeError):
            dur_sit_def = 0.0
    else:
        dur_sit_def = 0.0
    
    if 'duracion_localidad' in df.columns:
        dur_loc_def = df['duracion_localidad'].mode().iloc[0] if len(df['duracion_localidad'].mode()) > 0 else 0.0
        # Convertir a numérico si es necesario
        try:
            dur_loc_def = float(dur_loc_def)
        except (ValueError, TypeError):
            dur_loc_def = 0.0
    else:
        dur_loc_def = 0.0

    # Armar nuevo registro con valores por defecto (como en el notebook)
    Xnew = pd.DataFrame([{
        'situacion': situacion_sel,
        'servicio': servicio_sel,
        'sector_localidad': localidad_sel,
        'duracion_situacion': dur_sit_def,
        'duracion_localidad': dur_loc_def
    }])[FEATURES]

    # Realizar predicción automáticamente
    pred = model.predict(Xnew)[0]
    st.markdown(f"### 🧩 Predicción: **{pred}**")

    # Mostrar probabilidades si el modelo las tiene
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xnew)[0]
        proba_df = pd.DataFrame({
            'Categoría': getattr(model, "classes_", CATS),
            'Probabilidad': proba
        }).sort_values('Probabilidad', ascending=False)
        
        # Mostrar tabla con probabilidades
        st.dataframe(
            proba_df.assign(Probabilidad=lambda d: (d["Probabilidad"]*100).round(2).astype(str) + " %"),
            use_container_width=True,
            hide_index=True
        )
        
        # Mostrar gráfico de torta (como en el notebook)
        chart = (
            alt.Chart(proba_df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta('Probabilidad:Q', stack=True),
                color=alt.Color(
                    'Categoría:N',
                    scale=alt.Scale(
                        domain=CATS,
                        range=['#2ECC71', '#F1C40F', '#E74C3C']
                    ),
                    legend=alt.Legend(title="Categoría")
                ),
                tooltip=[
                    'Categoría:N',
                    alt.Tooltip('Probabilidad:Q', format='.2%', title='Probabilidad')
                ]
            )
            .properties(
                width=300,
                height=300,
                title="Distribución de Probabilidades de Predicción"
            )
        )
        st.altair_chart(chart, use_container_width=True)


import streamlit as st
import pandas as pd
import joblib
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Entrega 4 · Visualización e Integración", layout="wide")

# ---------------------------
# Config & helpers
# ---------------------------
DATA_FILE = "df_viz_for_streamlit.csv"
MODEL_FILE = "model.pkl"
FEATURES = ['situacion', 'servicio', 'sector_localidad', 'duracion_situacion', 'duracion_localidad']
TARGET = 'categoria_duracion'
CATS = ['Rápida', 'Normal', 'Lenta']  # mantener consistencia de colores/orden

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

# ---------------------------
# Layout superior
# ---------------------------
st.title("Entrega 4 — Visualización e Integración (Streamlit)")
st.caption("Explorá los datos, probá el modelo y comunicá los hallazgos con visualizaciones interactivas.")

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

# Sidebar: filtro global (opcional)
st.sidebar.header("Filtros rápidos")
servicios_all = ["(Todos)"] + sorted(df['servicio'].dropna().unique().tolist())
servicio_filter = st.sidebar.selectbox("Servicio (global)", options=servicios_all)
df_view = df.copy()
if servicio_filter != "(Todos)":
    df_view = df_view[df_view['servicio'] == servicio_filter]

st.sidebar.markdown(f"Registros visibles: **{len(df_view):,}**")

# ---------------------------
# Visualización 1: Torta por sector_localidad + leyenda %
# ---------------------------
st.markdown("## 🏙️ Distribución de categorías de duración por **localidad**")

sector_options = sorted(df_view['sector_localidad'].dropna().unique().tolist())
if not sector_options:
    st.info("No hay localidades disponibles con el filtro actual.")
else:
    sector_sel = st.selectbox("Seleccionar sector_localidad:", options=sector_options, key="sector")

    base_loc = (
        alt.Chart(df_view)
        .transform_filter(alt.datum.sector_localidad == sector_sel)
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

# ---------------------------
# Visualización 2: Torta por servicio + leyenda %
# ---------------------------
st.markdown("## ⚙️ Distribución de categorías de duración por **servicio**")

service_options = sorted(df_view['servicio'].dropna().unique().tolist())
if not service_options:
    st.info("No hay servicios disponibles con el filtro actual.")
else:
    service_sel = st.selectbox("Seleccionar servicio:", options=service_options, key="servicio_viz")

    base_srv = (
        alt.Chart(df_view)
        .transform_filter(alt.datum.servicio == service_sel)
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

# ---------------------------
# Visualización 3: Barras apiladas por situación (filtro por servicio)
# ---------------------------
st.markdown("## 📊 Proporciones de duración por **situación** (filtrado por servicio)")
servicios_all2 = sorted(df['servicio'].dropna().unique().tolist())
servicio_sel_bar = st.selectbox("Servicio:", options=servicios_all2, key="servicio_barras")

bars = (
    alt.Chart(df)
    .transform_filter(alt.datum.servicio == servicio_sel_bar)
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

# ---------------------------
# Simulador: predicción con menús dependientes
# ---------------------------
st.markdown("## 🎛️ Simulador de predicción")
if model is None:
    st.info("Subí **model.pkl** para habilitar el simulador.")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        servicio_sim = st.selectbox("Servicio (para filtrar situaciones):", options=sorted(df['servicio'].dropna().unique()))
        situaciones = sorted(df[df['servicio'] == servicio_sim]['situacion'].dropna().unique().tolist())
        situacion_sim = st.selectbox("Situación:", options=situaciones)
    with col_b:
        localidad_sim = st.selectbox("Localidad (sector_localidad):", options=sorted(df['sector_localidad'].dropna().unique()))

    # valores por defecto para num_features
    dur_sit_def = float(df['duracion_situacion'].median()) if 'duracion_situacion' in df.columns else 0.0
    dur_loc_def = float(df['duracion_localidad'].median()) if 'duracion_localidad' in df.columns else 0.0

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        dur_sit = st.number_input("duracion_situacion", min_value=0.0, value=dur_sit_def, step=1.0)
    with col_n2:
        dur_loc = st.number_input("duracion_localidad", min_value=0.0, value=dur_loc_def, step=1.0)

    if st.button("🔮 Predecir"):
        Xnew = pd.DataFrame([{
            'situacion': situacion_sim,
            'servicio': servicio_sim,
            'sector_localidad': localidad_sim,
            'duracion_situacion': dur_sit,
            'duracion_localidad': dur_loc
        }])[FEATURES]

        pred = model.predict(Xnew)[0]
        st.success(f"**Predicción:** {pred}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xnew)[0]
            proba_df = pd.DataFrame({"Categoría": getattr(model, "classes_", CATS), "Probabilidad": proba})\
                        .sort_values("Probabilidad", ascending=False)
            st.table(proba_df.assign(Probabilidad=lambda d: (d["Probabilidad"]*100).round(2).astype(str)+" %"))

st.caption("App basada en las visualizaciones y el modelo de la entrega 3/4.")

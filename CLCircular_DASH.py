# Librerías
import pdb
from pdb import pm as pdb_pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.datasets import grunfeld
from linearmodels.panel import PanelOLS
from linearmodels . panel import RandomEffects
from statsmodels.stats.diagnostic import het_breuschpagan
from linearmodels.panel import compare
from linearmodels.panel import PooledOLS
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import statistics as stats # estadística
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pylab import info, rcParams
from statsmodels.tsa.arima.model import ARIMA
from numpy._core.fromnumeric import transpose
import itertools
from semopy import Model, Optimizer, calc_stats
import openpyxl
import os
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Pharma Mexico vs USA",
    layout="wide"
)

st.title("Dashboard interactivo: México vs USA")
st.markdown("Basado en los dataframes derivados de `df_mexusa_mstj`.")

# -----------------------------
# Carga de datos
# -----------------------------
@st.cache_data
def load_data(file = '/Users/patoescamilla/Desktop/Files/Python/CLCircular - Datos/db1_internacional_pharma.xlsx'):
    df_mst = pd.read_excel(file, sheet_name="Mkt.Size", header=5)

    # Limpieza base
    df_mst = df_mst.dropna().copy()

    # Años como en tu script
    years = [str(year) for year in range(2011, 2026)]

    # Formato largo
    df_mstj = df_mst.melt(
        id_vars=["Geography", "Category", "Data Type", "Unit", "Current Constant"],
        value_vars=years,
        var_name="Year",
        value_name="Value"
    )

    # Conversión de tipos
    df_mstj["Year"] = pd.to_numeric(df_mstj["Year"], errors="coerce")
    df_mstj["Value"] = pd.to_numeric(df_mstj["Value"], errors="coerce")

    # Ajuste MXN -> USD, igual que en tu script
    df_mstj["Value"] = np.where(
        df_mstj["Unit"] == "MXN million",
        df_mstj["Value"] / 17.7,
        df_mstj["Value"]
    )

    df_mstj["Unit"] = "Million USD"

    # Categorías de interés como en tu código
    selected_categories = [
        "OTC",
        "Sports Nutrition",
        "Vitamins and Dietary Supplements",
        "Weight Management and Wellbeing",
        "Herbal/Traditional Products",
        "Allergy Care"
    ]

    df_prog = df_mstj[df_mstj["Category"].isin(selected_categories)].copy()

    # México y USA
    df_mexusa_mstj = df_prog[df_prog["Geography"].isin(["Mexico", "USA"])].copy()

    # Derivados
    df_plot = df_mexusa_mstj.copy()
    df_mex = df_mexusa_mstj[df_mexusa_mstj["Geography"] == "Mexico"].copy()
    df_usa = df_mexusa_mstj[df_mexusa_mstj["Geography"] == "USA"].copy()

    return df_mstj, df_prog, df_mexusa_mstj, df_plot, df_mex, df_usa


uploaded_file = st.file_uploader(
    "Sube el archivo Excel de market size",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Sube el archivo Excel para continuar.")
    st.stop()

df_mstj, df_prog, df_mexusa_mstj, df_plot, df_mex, df_usa = load_data(uploaded_file)

# -----------------------------
# Sidebar filtros
# -----------------------------
st.sidebar.header("Filtros")

available_geographies = sorted(df_mexusa_mstj["Geography"].dropna().unique().tolist())
available_categories = sorted(df_mexusa_mstj["Category"].dropna().unique().tolist())

selected_geographies = st.sidebar.multiselect(
    "Geografía",
    options=available_geographies,
    default=available_geographies
)

selected_categories = st.sidebar.multiselect(
    "Categoría",
    options=available_categories,
    default=available_categories
)

year_min = int(df_mexusa_mstj["Year"].min())
year_max = int(df_mexusa_mstj["Year"].max())

selected_years = st.sidebar.slider(
    "Rango de años",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max)
)

# -----------------------------
# Filtrado
# -----------------------------
df_filtered = df_mexusa_mstj[
    (df_mexusa_mstj["Geography"].isin(selected_geographies)) &
    (df_mexusa_mstj["Category"].isin(selected_categories)) &
    (df_mexusa_mstj["Year"].between(selected_years[0], selected_years[1]))
].copy()

if df_filtered.empty:
    st.warning("No hay datos con esos filtros.")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros", f"{len(df_filtered):,}")
col2.metric("Categorías", df_filtered["Category"].nunique())
col3.metric("Geografías", df_filtered["Geography"].nunique())
col4.metric("Valor total", f"{df_filtered['Value'].sum():,.2f}")

# -----------------------------
# Selector de vista
# -----------------------------
view_mode = st.radio(
    "Tipo de visualización",
    options=[
        "Comparativo México vs USA",
        "Solo México",
        "Solo USA"
    ],
    horizontal=True
)

if view_mode == "Solo México":
    df_view = df_filtered[df_filtered["Geography"] == "Mexico"].copy()
elif view_mode == "Solo USA":
    df_view = df_filtered[df_filtered["Geography"] == "USA"].copy()
else:
    df_view = df_filtered.copy()

if df_view.empty:
    st.warning("No hay datos para esa vista.")
    st.stop()

# -----------------------------
# Gráfica interactiva principal
# -----------------------------
st.subheader("Evolución del valor por categoría")

fig = px.line(
    df_view.sort_values(["Geography", "Category", "Year"]),
    x="Year",
    y="Value",
    color="Category",
    line_dash="Geography",
    markers=True,
    hover_data=["Geography", "Category", "Unit", "Data Type", "Current Constant"],
    title="Evolución del Valor por Categoría"
)

fig.update_layout(
    xaxis_title="Año",
    yaxis_title="Valor (Million USD)",
    legend_title="Categoría",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Resumen por año
# -----------------------------
st.subheader("Resumen anual")

summary_year = (
    df_view.groupby(["Year", "Geography", "Category"], as_index=False)["Value"]
    .sum()
    .sort_values(["Year", "Geography", "Category"])
)

st.dataframe(summary_year, use_container_width=True)

# -----------------------------
# Pivot table
# -----------------------------
st.subheader("Tabla pivote")

pivot_df = summary_year.pivot_table(
    index=["Geography", "Category"],
    columns="Year",
    values="Value",
    aggfunc="sum"
)

st.dataframe(pivot_df, use_container_width=True)

# -----------------------------
# Descarga
# -----------------------------
csv_data = df_view.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar datos filtrados (CSV)",
    data=csv_data,
    file_name="df_mexusa_mstj_filtrado.csv",
    mime="text/csv"
)


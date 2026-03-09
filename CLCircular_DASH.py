# Librerías
from ast import With
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
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

### Title and Page configuration

st.set_page_config(
    page_title="CLCircular - Dashboard",
    page_icon = "https://play-lh.googleusercontent.com/CMVIb6hsKkX7-4wlAZMfVOzVFbg6zCAI3MJFXtRbARXRhZbWHDLAKNSNuCOuM1i1gQ=w240-h480-rw",
    layout="wide"
)

st.image(
    "https://play-lh.googleusercontent.com/CMVIb6hsKkX7-4wlAZMfVOzVFbg6zCAI3MJFXtRbARXRhZbWHDLAKNSNuCOuM1i1gQ=w240-h480-rw",
    use_container_width=False
)

st.title("Dashboard de la Industria Farmacéutica")
st.markdown('Explorando diferentes estrategias de mercado')

tab_contexto, tab_exporta = st.tabs(["Contexto general", "¿Qué se exporta?"])

with tab_contexto:
    try:
        base_dir = Path(__file__).resolve().parent
        df_mst = pd.read_excel(base_dir / 'db1_internacional_pharma.xlsx', 'Mkt.Size', header=5)
        us_imports = pd.read_excel(base_dir / 'exportaciones_farmaceuticas_identificadas.xlsx')

        years = [str(year) for year in range(2011, 2026)]
        df_mstj = df_mst.melt(
            id_vars=['Geography', 'Category', 'Data Type', 'Unit', 'Current Constant'],
            value_vars=years,
            var_name='Year',
            value_name='Value'
        )
        df_mstj['Year'] = pd.to_numeric(df_mstj['Year'])
        df_mstj['Value'] = pd.to_numeric(df_mstj['Value'], errors='coerce')
        df_mstj['Value'] = np.where(df_mstj['Unit'] == 'MXN million', df_mstj['Value'] / 17.7, df_mstj['Value'])
        df_mstj['Unit'] = 'Million USD'

        selected_categories = [
            'OTC', 'Sports Nutrition', 'Vitamins and Dietary Supplements',
            'Weight Management and Wellbeing', 'Herbal/Traditional Products', 'Allergy Care'
        ]
        df_prog = df_mstj[df_mstj['Category'].isin(selected_categories)]
        df_usa = df_prog[df_prog['Geography'] == 'USA']

        st.success("Datos cargados correctamente")

        st.sidebar.header("Filtros")
        view_mode = st.sidebar.radio(
            "Vista de gráfica",
            ["Gráfica normal", "Modelos SARIMA"]
        )

        min_year = int(df_usa['Year'].min())
        max_year = int(df_usa['Year'].max())
        year_filter = st.sidebar.slider(
            "Rango de años",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )

        sarima_categories = ['OTC', 'Sports Nutrition', 'Vitamins and Dietary Supplements']

        df_usa_year_filtered = df_usa[(df_usa['Year'] >= year_filter[0]) & (df_usa['Year'] <= year_filter[1])]

        with st.container():
            if view_mode == "Gráfica normal":
                categories = sorted(df_usa_year_filtered['Category'].dropna().unique().tolist())
                category_filter = st.multiselect(
                    "Filtra categorías",
                    options=categories,
                    default=categories
                )
                df_usa_filtered = df_usa_year_filtered[df_usa_year_filtered['Category'].isin(category_filter)].sort_values(['Category', 'Year'])

                if df_usa_filtered.empty:
                    st.warning("Selecciona al menos una categoría para mostrar la gráfica.")
                else:
                    fig = px.line(
                        df_usa_filtered,
                        x='Year',
                        y='Value',
                        color='Category',
                        title='Tamaño del Mercado Farmacéutico Internacional',
                        markers=True
                    )
                    share_normal_df = (
                        df_usa_filtered
                        .groupby('Category', as_index=False)['Value']
                        .sum()
                    )

                    fig_share_normal = px.pie(
                        share_normal_df,
                        names='Category',
                        values='Value',
                        hole=0.35,
                        title=f'Share por Categoría ({year_filter[0]}-{year_filter[1]})'
                    )

                    col_left, col_right = st.columns([2.2, 1.2])
                    with col_left:
                        st.plotly_chart(fig, use_container_width=True)
                    with col_right:
                        st.plotly_chart(fig_share_normal, use_container_width=True)
            else:
                sarima_filter = st.multiselect(
                    "Filtra categorías SARIMA",
                    options=sarima_categories,
                    default=sarima_categories
                )
                forecast_steps = st.slider("Años de pronóstico (SARIMA)", min_value=1, max_value=10, value=5)
                series_data = []
                color_map = {
                    'OTC': '#1f77b4',
                    'Sports Nutrition': '#2ca02c',
                    'Vitamins and Dietary Supplements': '#d62728'
                }

                if not sarima_filter:
                    st.warning("Selecciona al menos una categoría para modelar con SARIMA.")
                    sarima_filter = []

                for category in sarima_filter:
                    cat_df = (
                        df_usa_year_filtered[df_usa_year_filtered['Category'] == category][['Year', 'Value']]
                        .dropna()
                        .sort_values('Year')
                    )
                    if cat_df.empty or len(cat_df) < 8:
                        st.warning(f"No hay suficientes datos para SARIMA en {category}.")
                        continue

                    series = cat_df.set_index('Year')['Value'].astype(float)

                    try:
                        model = SARIMAX(
                            series,
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 4),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        results = model.fit(disp=False)
                        forecast_obj = results.get_forecast(steps=forecast_steps)
                        forecast = forecast_obj.predicted_mean
                        conf_int = forecast_obj.conf_int(alpha=0.05)
                        lower = conf_int.iloc[:, 0].values
                        upper = conf_int.iloc[:, 1].values

                        forecast_years = np.arange(
                            int(series.index.max()) + 1,
                            int(series.index.max()) + 1 + forecast_steps
                        )

                        observed_df = cat_df.copy().sort_values('Year')
                        forecast_df = pd.DataFrame({'Year': forecast_years, 'Value': forecast.values})
                        ci_df = pd.DataFrame({'Year': forecast_years, 'Lower': lower, 'Upper': upper})

                        series_data.append((category, observed_df, forecast_df, ci_df))
                    except Exception as sarima_error:
                        st.warning(f"No se pudo ajustar SARIMA para {category}: {sarima_error}")

                if not series_data:
                    st.error("No se pudo generar la gráfica SARIMA con las categorías seleccionadas.")
                else:
                    fig_sarima = go.Figure()

                    for category, observed_df, forecast_df, ci_df in series_data:
                        color = color_map.get(category, '#636EFA')

                        fig_sarima.add_trace(go.Scatter(
                            x=observed_df['Year'],
                            y=observed_df['Value'],
                            mode='lines+markers',
                            name=category,
                            line={'color': color, 'width': 2},
                            legendgroup=category
                        ))

                        fig_sarima.add_trace(go.Scatter(
                            x=forecast_df['Year'],
                            y=forecast_df['Value'],
                            mode='lines+markers',
                            name=f'{category} - SARIMA',
                            line={'color': color, 'width': 2, 'dash': 'dash'},
                            showlegend=False,
                            legendgroup=category
                        ))

                        fig_sarima.add_trace(go.Scatter(
                            x=ci_df['Year'],
                            y=ci_df['Upper'],
                            mode='lines',
                            line={'width': 0},
                            hoverinfo='skip',
                            showlegend=False,
                            legendgroup=category
                        ))

                        fig_sarima.add_trace(go.Scatter(
                            x=ci_df['Year'],
                            y=ci_df['Lower'],
                            mode='lines',
                            line={'width': 0},
                            fill='tonexty',
                            fillcolor='rgba(31, 119, 180, 0.12)' if category == 'OTC' else (
                                'rgba(44, 160, 44, 0.12)' if category == 'Sports Nutrition' else 'rgba(214, 39, 40, 0.12)'
                            ),
                            name=f'{category} - IC 95%',
                            showlegend=False,
                            legendgroup=category
                        ))

                    fig_sarima.update_layout(
                        title='Modelos SARIMA - Categorías Seleccionadas',
                        xaxis_title='Año',
                        yaxis_title='Valor',
                        hovermode='x unified'
                    )

                    share_records = []
                    for category, observed_df, forecast_df, ci_df in series_data:
                        if not forecast_df.empty:
                            target_year = int(forecast_df['Year'].max())
                            target_value = float(forecast_df['Value'].sum())
                        else:
                            target_year = int(observed_df['Year'].max())
                            target_value = float(observed_df['Value'].sum())

                        share_records.append({
                            'Category': category,
                            'Year': target_year,
                            'Value': target_value
                        })

                    share_df = pd.DataFrame(share_records)
                    latest_year = int(share_df['Year'].max())

                    fig_share = px.pie(
                        share_df,
                        names='Category',
                        values='Value',
                        hole=0.35,
                        title=f'Share SARIMA por Categoría (Forecast acumulado, base {year_filter[0]}-{year_filter[1]})'
                    )

                    col_left, col_right = st.columns([2.2, 1.2])
                    with col_left:
                        st.plotly_chart(fig_sarima, use_container_width=True)
                    with col_right:
                        st.plotly_chart(fig_share, use_container_width=True)

        st.markdown("---")
        st.subheader("Valor Total por Subsegmento (USA)")

        subsegment_categories = [
            'Adult Mouth Care', 'Analgesics', 'Sleep Aids',
            'Cough, Cold and Allergy (Hay Fever) Remedies', 'Dermatologicals',
            'Digestive Remedies', 'Emergency Contraception', 'Eye Care',
            'NRT Smoking Cessation Aids', 'Wound Care'
        ]
        subsegment_options = st.multiselect(
            "Subsegmentos",
            options=["Todos"] + subsegment_categories,
            default=["Todos"]
        )
        if (not subsegment_options) or ("Todos" in subsegment_options):
            subsegment_filter = subsegment_categories
        else:
            subsegment_filter = subsegment_options

        df_subsegments = df_mstj[
            (df_mstj['Geography'] == 'USA') &
            (df_mstj['Year'] >= year_filter[0]) &
            (df_mstj['Year'] <= year_filter[1]) &
            (df_mstj['Category'].isin(subsegment_filter))
        ].dropna(subset=['Value'])

        if df_subsegments.empty:
            st.warning("No hay datos disponibles para los subsegmentos seleccionados.")
        else:
            df_subsegments_trend = (
                df_subsegments
                .groupby(['Year', 'Category'], as_index=False)['Value']
                .sum()
                .sort_values(['Category', 'Year'])
            )

            if view_mode == "Gráfica normal":
                fig_subsegments_trend = px.line(
                    df_subsegments_trend,
                    x='Year',
                    y='Value',
                    color='Category',
                    markers=True,
                    title='Evolución de Subsegmentos a lo Largo del Tiempo',
                    labels={'Value': 'Valor (Million USD)', 'Year': 'Año', 'Category': 'Subsegmento'}
                )
                fig_subsegments_trend.update_yaxes(tickprefix="$", separatethousands=True)
                st.plotly_chart(fig_subsegments_trend, use_container_width=True)
            else:
                subsegment_forecast_steps = st.slider(
                    "Años de pronóstico SARIMA (Subsegmentos)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="subsegment_forecast_steps"
                )
                subsegment_sarima_params = {
                    'Cough, Cold and Allergy (Hay Fever) Remedies': ((0, 1, 1), (1, 1, 1, 12)),
                    'Analgesics': ((0, 1, 1), (1, 1, 1, 12)),
                    'Digestive Remedies': ((0, 1, 1), (1, 1, 1, 12))
                }
                subsegment_fill_colors = {
                    'Cough, Cold and Allergy (Hay Fever) Remedies': 'rgba(31, 119, 180, 0.12)',
                    'Analgesics': 'rgba(44, 160, 44, 0.12)',
                    'Digestive Remedies': 'rgba(214, 39, 40, 0.12)'
                }
                palette = px.colors.qualitative.Plotly
                fig_subsegments_sarima = go.Figure()
                modeled_count = 0

                for idx, category in enumerate(subsegment_filter):
                    if category not in subsegment_sarima_params:
                        continue

                    cat_df = (
                        df_subsegments_trend[df_subsegments_trend['Category'] == category][['Year', 'Value']]
                        .dropna()
                        .sort_values('Year')
                    )
                    if cat_df.empty or len(cat_df) < 8:
                        st.warning(f"No hay suficientes datos para SARIMA en {category}.")
                        continue

                    series = cat_df.set_index('Year')['Value'].astype(float)
                    order, seasonal_order = subsegment_sarima_params[category]

                    try:
                        model = SARIMAX(
                            series,
                            order=order,
                            seasonal_order=seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        results = model.fit(disp=False)
                        forecast_obj = results.get_forecast(steps=subsegment_forecast_steps)
                        forecast = forecast_obj.predicted_mean
                        conf_int = forecast_obj.conf_int(alpha=0.05)
                        forecast_years = np.arange(
                            int(series.index.max()) + 1,
                            int(series.index.max()) + 1 + subsegment_forecast_steps
                        )
                        color = palette[idx % len(palette)]

                        fig_subsegments_sarima.add_trace(go.Scatter(
                            x=cat_df['Year'],
                            y=cat_df['Value'],
                            mode='lines+markers',
                            name=category,
                            line={'color': color, 'width': 2},
                            legendgroup=category
                        ))
                        fig_subsegments_sarima.add_trace(go.Scatter(
                            x=forecast_years,
                            y=forecast.values,
                            mode='lines+markers',
                            name=f'{category} - SARIMA',
                            line={'color': color, 'width': 2, 'dash': 'dash'},
                            showlegend=False,
                            legendgroup=category
                        ))
                        fig_subsegments_sarima.add_trace(go.Scatter(
                            x=forecast_years,
                            y=conf_int.iloc[:, 0].values,
                            mode='lines',
                            line={'width': 0},
                            hoverinfo='skip',
                            showlegend=False,
                            legendgroup=category
                        ))
                        fig_subsegments_sarima.add_trace(go.Scatter(
                            x=forecast_years,
                            y=conf_int.iloc[:, 1].values,
                            mode='lines',
                            line={'width': 0},
                            fill='tonexty',
                            fillcolor=subsegment_fill_colors.get(category, 'rgba(31, 119, 180, 0.12)'),
                            hoverinfo='skip',
                            showlegend=False,
                            legendgroup=category
                        ))
                        modeled_count += 1
                    except Exception as sarima_error:
                        st.warning(f"No se pudo ajustar SARIMA para {category}: {sarima_error}")

                if modeled_count == 0:
                    st.error("No se pudo generar la evolución SARIMA de subsegmentos con la selección actual.")
                else:
                    fig_subsegments_sarima.update_layout(
                        title='Evolución de Subsegmentos con Modelos SARIMA',
                        xaxis_title='Año',
                        yaxis_title='Valor',
                        hovermode='x unified'
                    )
                    fig_subsegments_sarima.update_yaxes(tickprefix="$", separatethousands=True)
                    st.plotly_chart(fig_subsegments_sarima, use_container_width=True)

            df_subsegments_total = (
                df_subsegments
                .groupby('Category', as_index=False)['Value']
                .sum()
                .sort_values('Value', ascending=True)
            )

            fig_subsegments = px.bar(
                df_subsegments_total,
                x='Value',
                y='Category',
                orientation='h',
                title='Total Acumulado por Subsegmento',
                labels={'Value': 'Valor Total (Million USD)', 'Category': 'Subsegmento'},
                color='Value',
                color_continuous_scale='Blues'
            )
            fig_subsegments.update_traces(
                text=df_subsegments_total['Value'].map(lambda x: f"${x:,.1f}"),
                textposition='inside',
                insidetextanchor='middle'
            )
            fig_subsegments.update_xaxes(tickprefix="$", separatethousands=True)
            fig_subsegments.update_layout(coloraxis_showscale=False, height=520)

            fig_subsegments_treemap = px.treemap(
                df_subsegments_total,
                path=['Category'],
                values='Value',
                title='Share Acumulado por Subsegmento'
            )
            fig_subsegments_treemap.update_traces(textinfo='label+percent entry')
            fig_subsegments_treemap.update_layout(height=520)

            col_bar, col_tree = st.columns([1.9, 1.1], gap="small")
            with col_bar:
                st.plotly_chart(fig_subsegments, use_container_width=True)
            with col_tree:
                st.plotly_chart(fig_subsegments_treemap, use_container_width=True)

    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")

with tab_exporta:
    st.subheader("¿Qué se exporta?")
    try:
        base_dir = Path(__file__).resolve().parent
        us_imports = pd.read_excel(base_dir / 'exportaciones_farmaceuticas_identificadas.xlsx')

        st.sidebar.markdown("---")
        st.sidebar.subheader("¿Qué se exporta?")
        export_years = sorted(us_imports['Year'].dropna().astype(int).unique().tolist())
        selected_export_year = st.sidebar.selectbox(
            "Año de exportación",
            options=export_years,
            index=len(export_years) - 1,
            key="selected_export_year"
        )
        export_chapters = sorted(us_imports['Chapter 4 Digit'].dropna().unique().tolist())
        selected_export_chapters = st.sidebar.multiselect(
            "Capítulos",
            options=export_chapters,
            default=export_chapters,
            key="selected_export_chapters"
        )
        export_hs2 = sorted(us_imports['HS2 4 Digit'].dropna().unique().tolist())
        selected_export_hs2 = st.sidebar.multiselect(
            "HS2 4 Digit",
            options=export_hs2,
            default=export_hs2,
            key="selected_export_hs2"
        )
        export_months = sorted(us_imports['Month'].dropna().astype(str).unique().tolist())
        selected_export_months = st.sidebar.multiselect(
            "Mes",
            options=export_months,
            default=export_months,
            key="selected_export_months"
        )

        df_exports_filtered = us_imports[us_imports['Year'].astype(int) == selected_export_year].copy()
        if not selected_export_chapters or not selected_export_hs2 or not selected_export_months:
            df_exports_filtered = df_exports_filtered.iloc[0:0]
        else:
            df_exports_filtered = df_exports_filtered[
                df_exports_filtered['Chapter 4 Digit'].isin(selected_export_chapters)
                & df_exports_filtered['HS2 4 Digit'].isin(selected_export_hs2)
                & df_exports_filtered['Month'].astype(str).isin(selected_export_months)
            ]

        if df_exports_filtered.empty:
            st.warning("No hay datos de exportación con los filtros seleccionados.")
        else:
            total_trade_value = float(df_exports_filtered['Trade Value'].sum())
            st.metric("Valor total exportado", f"${total_trade_value:,.0f}")

            df_hs4_hover = (
                df_exports_filtered
                .groupby('HS2 4 Digit')['HS4 4 Digit']
                .apply(
                    lambda s: "<br>".join(sorted(pd.Series(s).dropna().astype(str).unique()[:8])) +
                    ("<br>..." if len(pd.Series(s).dropna().astype(str).unique()) > 8 else "")
                )
                .reset_index(name='HS4_hover')
            )

            df_top_exports = (
                df_exports_filtered
                .groupby('HS2 4 Digit', as_index=False)['Trade Value']
                .sum()
                .merge(df_hs4_hover, on='HS2 4 Digit', how='left')
                .sort_values('Trade Value', ascending=False)
                .head(15)
            )
            fig_exports = px.bar(
                df_top_exports.sort_values('Trade Value', ascending=True),
                x='Trade Value',
                y='HS2 4 Digit',
                orientation='h',
                title=f'Top exportaciones ({selected_export_year})',
                labels={'Trade Value': 'Trade Value', 'HS2 4 Digit': 'Producto HS2'},
                color='Trade Value',
                color_continuous_scale='Tealgrn',
                custom_data=['HS4_hover']
            )
            fig_exports.update_traces(
                hovertemplate="<b>%{y}</b><br>Trade Value: $%{x:,.0f}<br><br><b>HS4 4 Digit</b><br>%{customdata[0]}<extra></extra>"
            )
            fig_exports.update_xaxes(tickprefix="$", separatethousands=True)
            fig_exports.update_layout(coloraxis_showscale=False, height=520)
            st.plotly_chart(fig_exports, use_container_width=True)

            st.dataframe(
                df_exports_filtered[['Year', 'Chapter 4 Digit', 'HS2 4 Digit', 'HS4 4 Digit', 'Trade Value', 'Share']],
                use_container_width=True,
                hide_index=True
            )
    except Exception as e:
        st.error(f"Error al cargar exportaciones: {e}")

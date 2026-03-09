# Librerías
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX


def kmeans_2d(points: np.ndarray, k: int, max_iter: int = 100, seed: int = 42):
    if len(points) == 0:
        return np.array([], dtype=int), np.empty((0, 2))
    k = min(k, len(points))
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(len(points), size=k, replace=False)
    centroids = points[init_idx].astype(float)
    labels = np.zeros(len(points), dtype=int)

    for _ in range(max_iter):
        dist = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            cluster_points = points[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = cluster_points.mean(axis=0)
            else:
                centroids[i] = points[rng.integers(0, len(points))]
    return labels, centroids


def assign_mexico_regions(coords_df: pd.DataFrame) -> pd.DataFrame:
    result = coords_df.copy()
    if result.empty:
        result['region_cluster'] = pd.Series(dtype=str)
        return result

    points = result[['latitud_aprox', 'longitud_aprox']].to_numpy(dtype=float)
    labels, centroids = kmeans_2d(points, k=4)
    result['cluster_id'] = labels

    centroids_df = pd.DataFrame(centroids, columns=['lat', 'lon'])
    centroids_df['cluster_id'] = centroids_df.index

    cluster_norte = int(centroids_df.loc[centroids_df['lat'].idxmax(), 'cluster_id'])
    cluster_sur = int(centroids_df.loc[centroids_df['lat'].idxmin(), 'cluster_id'])
    remaining = centroids_df[
        ~centroids_df['cluster_id'].isin([cluster_norte, cluster_sur])
    ].copy()

    cluster_golfo = None
    cluster_centro = None
    if not remaining.empty:
        cluster_golfo = int(remaining.loc[remaining['lon'].idxmax(), 'cluster_id'])
        cluster_centro = int(remaining.loc[remaining['lon'].idxmin(), 'cluster_id'])

    cluster_to_region = {
        cluster_norte: 'Región Norte',
        cluster_sur: 'Región Sur'
    }
    if cluster_golfo is not None:
        cluster_to_region[cluster_golfo] = 'Región Golfo'
    if cluster_centro is not None:
        cluster_to_region[cluster_centro] = 'Región Centro'

    result['region_cluster'] = result['cluster_id'].map(cluster_to_region).fillna('Región Golfo')
    return result.drop(columns=['cluster_id'])


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
        em_int = pd.read_excel(base_dir / 'exportaciones_equipo_medico_empresas_sectores_urls.xlsx')
        hs2_excluded = {
            'Preparaciones Alimenticias Diversas',
            'Instrumentos Musicales; sus Partes y Accesorios',
            'Jabones, Lubricantes, Ceras, Velas, Pastas de Modelar',
            'Aparatos de Relojería y sus Partes'
        }
        us_imports = us_imports[~us_imports['HS2 4 Digit'].astype(str).isin(hs2_excluded)].copy()
        em_int = em_int[~em_int['HS2 4 Digit'].astype(str).isin(hs2_excluded)].copy()
        plantas_file = base_dir / 'empresas_con_plantas_en_mexico_y_exportadores_MX_USA.xlsx'
        plantas_80_raw = pd.read_excel(plantas_file, sheet_name='Producto_80', header=3)
        plantas_20_raw = pd.read_excel(plantas_file, sheet_name='Exportadores_20', header=3)

        st.sidebar.markdown("---")
        st.sidebar.subheader("¿Qué se exporta?")
        export_mode = st.sidebar.radio(
            "Vista de datos",
            ["Todo", "Farmacéutica", "Maquinaria"],
            key="export_mode_selector"
        )
        include_pharma = export_mode in ["Farmacéutica", "Todo"]
        include_maquinaria = export_mode in ["Maquinaria", "Todo"]

        year_sources = []
        if include_pharma:
            year_sources.append(us_imports['Year'])
        if include_maquinaria:
            year_sources.append(em_int['Year'])
        export_years = sorted(
            pd.concat(year_sources, ignore_index=True).dropna().astype(int).unique().tolist()
        )
        selected_export_year_range = st.sidebar.slider(
            "Rango de años de exportación",
            min_value=min(export_years),
            max_value=max(export_years),
            value=(min(export_years), max(export_years)),
            key="selected_export_year_range"
        )
        sector_hs2_map = {
            'Belleza / maquillaje / cuidado de la piel': [
                'Aceites Esenciales, Perfumes, Cosméticos, Artículos de Tocador'
            ],
            'Higiene bucal / dental': [
                'Aceites Esenciales, Perfumes, Cosméticos, Artículos de Tocador',
                'Jabones, Lubricantes, Ceras, Velas, Pastas de Modelar'
            ],
            'Levaduras / microorganismos': [
                'Preparaciones Alimenticias Diversas',
                'Productos Diversos de las Industrias Químicas'
            ],
            'Medicamentos dosificados (HS 3004)': [
                'Productos Farmacéuticos'
            ],
            'Medicamentos no dosificados / APIs (HS 3003 y afines)': [
                'Productos Farmacéuticos',
                'Productos Químicos Orgánicos'
            ],
            'Medios de cultivo / diagnóstico microbiológico': [
                'Productos Diversos de las Industrias Químicas',
                'Productos Farmacéuticos'
            ],
            'Preparaciones dentales / impresión dental': [
                'Productos Farmacéuticos',
                'Productos Diversos de las Industrias Químicas'
            ],
            'Vitaminas / suplementos / químicos nutricionales': [
                'Productos Farmacéuticos',
                'Preparaciones Alimenticias Diversas',
                'Productos Químicos Orgánicos'
            ]
        }
        sector_hs2_map = {
            sector: [hs2 for hs2 in hs2_list if hs2 not in hs2_excluded]
            for sector, hs2_list in sector_hs2_map.items()
        }
        hs2_with_data = set()
        if include_pharma:
            hs2_with_data |= set(us_imports['HS2 4 Digit'].dropna().astype(str).unique().tolist())
        if include_maquinaria:
            hs2_with_data |= set(em_int['HS2 4 Digit'].dropna().astype(str).unique().tolist())
        hs2_from_sector_map_all = {
            hs2 for hs2_list in sector_hs2_map.values() for hs2 in hs2_list
        }
        hs2_from_em_all = (
            set(em_int['HS2 4 Digit'].dropna().astype(str).unique().tolist())
            if include_maquinaria else set()
        )
        hs2_relevant = hs2_with_data.intersection(hs2_from_sector_map_all.union(hs2_from_em_all))
        export_hs2 = sorted(hs2_relevant if hs2_relevant else hs2_with_data)

        selected_export_hs2 = st.sidebar.multiselect(
            "HS2 4 Digit",
            options=["Todos"] + export_hs2,
            default=["Todos"],
            key="selected_export_hs2"
        )
        sectores_presentes = set()
        if include_pharma:
            sectores_presentes |= set(plantas_80_raw['grupo_producto'].dropna().astype(str).unique().tolist())
        if include_maquinaria:
            sectores_presentes |= set(em_int['sector_grupo'].dropna().astype(str).unique().tolist())
        sectores_presentes = sorted(sectores_presentes)
        selected_sectores = st.sidebar.multiselect(
            f"Sectores ({len(sectores_presentes)} presentes)",
            options=["Todos"] + sectores_presentes,
            default=["Todos"],
            key="selected_sectores_mapa"
        )
        selected_region_cluster = st.sidebar.radio(
            "Región (clúster)",
            ["Todo", "Región Norte", "Región Sur", "Región Golfo", "Región Centro"],
            key="selected_region_cluster"
        )
        if (not selected_export_hs2) or ("Todos" in selected_export_hs2):
            selected_export_hs2 = export_hs2
        if (not selected_sectores) or ("Todos" in selected_sectores):
            selected_sectores = sectores_presentes

        hs2_from_sectors = sorted({
            hs2
            for sector in selected_sectores
            for hs2 in sector_hs2_map.get(sector, [])
        })
        if include_maquinaria:
            hs2_from_em_sectors = sorted(
                em_int[em_int['sector_grupo'].astype(str).isin(selected_sectores)]['HS2 4 Digit']
                .dropna().astype(str).unique().tolist()
            )
        else:
            hs2_from_em_sectors = []
        hs2_pool = sorted(set(hs2_from_sectors).union(set(hs2_from_em_sectors)))
        selected_hs2_effective = sorted(set(selected_export_hs2).intersection(hs2_pool))

        sector_share_totals = (
            plantas_80_raw
            .groupby('grupo_producto', as_index=False)['market_share_aprox_pct']
            .sum()
        )
        sector_share_map = {
            row['grupo_producto']: float(row['market_share_aprox_pct']) / 100.0
            for _, row in sector_share_totals.iterrows()
        }
        sector_hs2_rows = []
        for sector_name, hs2_list in sector_hs2_map.items():
            sector_share = sector_share_map.get(sector_name, 0.0)
            if not hs2_list:
                continue
            per_hs2_weight = sector_share / len(hs2_list)
            for hs2_name in hs2_list:
                sector_hs2_rows.append({
                    'sector': sector_name,
                    'HS2 4 Digit': hs2_name,
                    'weight': per_hs2_weight
                })
        sector_hs2_weights = pd.DataFrame(sector_hs2_rows)
        hs2_weight_total = sector_hs2_weights.groupby('HS2 4 Digit', as_index=False)['weight'].sum()
        hs2_weight_selected = (
            sector_hs2_weights[sector_hs2_weights['sector'].isin(selected_sectores)]
            .groupby('HS2 4 Digit', as_index=False)['weight']
            .sum()
        )
        hs2_weight_factors = hs2_weight_total.merge(
            hs2_weight_selected,
            on='HS2 4 Digit',
            how='left',
            suffixes=('_total', '_selected')
        ).fillna({'weight_selected': 0.0})
        hs2_weight_factors['sector_multiplier'] = np.where(
            hs2_weight_factors['weight_total'] > 0,
            hs2_weight_factors['weight_selected'] / hs2_weight_factors['weight_total'],
            0.0
        )
        hs2_multiplier_map = {
            row['HS2 4 Digit']: float(row['sector_multiplier'])
            for _, row in hs2_weight_factors.iterrows()
        }
        if include_pharma:
            df_exports_base = us_imports[
                (us_imports['Year'].astype(int) >= selected_export_year_range[0]) &
                (us_imports['Year'].astype(int) <= selected_export_year_range[1]) &
                (us_imports['HS2 4 Digit'].astype(str).isin(selected_hs2_effective))
            ].copy()
        else:
            df_exports_base = us_imports.iloc[0:0].copy()

        if include_maquinaria:
            em_base = em_int[
                (em_int['Year'].astype(int) >= selected_export_year_range[0]) &
                (em_int['Year'].astype(int) <= selected_export_year_range[1]) &
                (em_int['HS2 4 Digit'].astype(str).isin(selected_hs2_effective))
            ].copy()
        else:
            em_base = em_int.iloc[0:0].copy()

        export_months = sorted(
            pd.concat(
                [df_exports_base['Month'].dropna().astype(str), em_base['Month'].dropna().astype(str)],
                ignore_index=True
            ).unique().tolist()
        )
        if export_months:
            if len(export_months) == 1:
                selected_month_range = (export_months[0], export_months[0])
                st.caption(f"Mes seleccionado: {export_months[0]}")
            else:
                selected_month_range = st.select_slider(
                    "Rango de meses",
                    options=export_months,
                    value=(export_months[0], export_months[-1]),
                    key="selected_export_month_range"
                )
            start_month, end_month = selected_month_range
            df_exports_filtered = df_exports_base[
                (df_exports_base['Month'].astype(str) >= start_month)
                & (df_exports_base['Month'].astype(str) <= end_month)
            ]
        else:
            df_exports_filtered = df_exports_base.iloc[0:0]
        if export_months:
            em_filtered = em_base[
                (em_base['Month'].astype(str) >= start_month) &
                (em_base['Month'].astype(str) <= end_month)
            ].copy()
        else:
            em_filtered = em_base.iloc[0:0].copy()

        df_exports_filtered = df_exports_filtered.copy()
        if not df_exports_filtered.empty:
            df_exports_filtered['sector_multiplier'] = df_exports_filtered['HS2 4 Digit'].map(hs2_multiplier_map).fillna(0.0)
            df_exports_filtered['Trade Value Ajustado'] = df_exports_filtered['Trade Value'] * df_exports_filtered['sector_multiplier']
        else:
            df_exports_filtered = df_exports_filtered.assign(
                sector_multiplier=pd.Series(dtype=float),
                **{'Trade Value Ajustado': pd.Series(dtype=float)}
            )
        if not em_filtered.empty:
            em_filtered = em_filtered.copy()
            em_filtered['Trade Value Ajustado'] = em_filtered['Trade Value']
        else:
            em_filtered = em_filtered.assign(**{'Trade Value Ajustado': pd.Series(dtype=float)})

        # La barra de HS2 integra us_imports + em_int.
        # KPIs y tabla izquierda se mantienen con el flujo base (us_imports).
        bar_sources = [df_exports_filtered[['HS2 4 Digit', 'HS4 4 Digit', 'Trade Value Ajustado']]]
        if not em_filtered.empty:
            bar_sources.append(em_filtered[['HS2 4 Digit', 'HS4 4 Digit', 'Trade Value Ajustado']])
        df_bar_filtered = pd.concat(bar_sources, ignore_index=True)
        df_kpi_filtered = pd.concat(
            [df_exports_filtered[['Month', 'Trade Value Ajustado']], em_filtered[['Month', 'Trade Value Ajustado']]],
            ignore_index=True
        )

        if df_kpi_filtered.empty:
            st.warning("No hay datos de exportación con los filtros seleccionados.")
        else:
            total_trade_value = float(df_kpi_filtered['Trade Value Ajustado'].sum())
            monthly_totals = (
                df_kpi_filtered.assign(
                    period=pd.to_datetime(df_kpi_filtered['Month'].astype(str) + '-01', errors='coerce')
                )
                .dropna(subset=['period'])
                .groupby('period', as_index=False)['Trade Value Ajustado']
                .sum()
                .sort_values('period')
            )

            growth_pct_display = "N/A"
            growth_range_label = ""
            if not monthly_totals.empty:
                first_period = monthly_totals.iloc[0]['period']
                last_period = monthly_totals.iloc[-1]['period']
                first_value = float(monthly_totals.iloc[0]['Trade Value Ajustado'])
                last_value = float(monthly_totals.iloc[-1]['Trade Value Ajustado'])
                growth_range_label = f"{first_period.strftime('%Y-%m')} → {last_period.strftime('%Y-%m')}"
                if first_value != 0:
                    growth_pct = ((last_value - first_value) / first_value) * 100
                    growth_pct_display = f"{growth_pct:,.1f}%"

            kpi_col_1, kpi_col_2 = st.columns(2)
            with kpi_col_1:
                st.metric("Valor total exportado", f"${total_trade_value:,.0f}")
            with kpi_col_2:
                st.metric("Crecimiento % (inicio-fin)", growth_pct_display)
                if growth_range_label:
                    st.caption(growth_range_label)

            df_hs4_hover = (
                df_bar_filtered
                .groupby('HS2 4 Digit')['HS4 4 Digit']
                .apply(
                    lambda s: "<br>".join(sorted(pd.Series(s).dropna().astype(str).unique()[:8])) +
                    ("<br>..." if len(pd.Series(s).dropna().astype(str).unique()) > 8 else "")
                )
                .reset_index(name='HS4_hover')
            )

            df_top_exports = (
                df_bar_filtered
                .groupby('HS2 4 Digit', as_index=False)['Trade Value Ajustado']
                .sum()
                .merge(df_hs4_hover, on='HS2 4 Digit', how='left')
                .sort_values('Trade Value Ajustado', ascending=False)
                .head(15)
            )
            fig_exports = px.bar(
                df_top_exports.sort_values('Trade Value Ajustado', ascending=True),
                x='Trade Value Ajustado',
                y='HS2 4 Digit',
                orientation='h',
                title=f'Top exportaciones ({selected_export_year_range[0]}-{selected_export_year_range[1]})',
                labels={'Trade Value Ajustado': 'Trade Value Ajustado', 'HS2 4 Digit': 'Producto HS2'},
                color='Trade Value Ajustado',
                color_continuous_scale='Tealgrn',
                custom_data=['HS4_hover']
            )
            df_top_exports_plot = df_top_exports.sort_values('Trade Value Ajustado', ascending=True).copy()
            fig_exports.update_traces(
                text=df_top_exports_plot['Trade Value Ajustado'].map(lambda x: f"${x:,.0f}"),
                textposition='outside',
                insidetextanchor='end',
                hovertemplate="<b>%{y}</b><br>Trade Value: $%{x:,.0f}<br><br><b>HS4 4 Digit</b><br>%{customdata[0]}<extra></extra>"
            )
            fig_exports.update_xaxes(tickprefix="$", separatethousands=True)
            fig_exports.update_layout(coloraxis_showscale=False, height=520, uniformtext_minsize=9, uniformtext_mode='hide')
            st.plotly_chart(fig_exports, use_container_width=True)

            sectors_from_hs2 = {
                sector
                for sector, hs2_list in sector_hs2_map.items()
                if any(hs2 in selected_hs2_effective for hs2 in hs2_list)
            }
            selected_sectores_effective = sorted(set(selected_sectores).intersection(sectors_from_hs2))

            if selected_sectores_effective:
                plantas_80 = plantas_80_raw[
                    plantas_80_raw['grupo_producto'].astype(str).isin(selected_sectores_effective)
                ].copy()
            else:
                plantas_80 = plantas_80_raw.iloc[0:0].copy()
            empresas_sectores_activas = set(plantas_80['empresa'].dropna().astype(str).unique())
            if selected_hs2_effective and empresas_sectores_activas:
                plantas_20 = plantas_20_raw[
                    plantas_20_raw['empresa'].astype(str).isin(empresas_sectores_activas)
                ].copy()
            else:
                plantas_20 = plantas_20_raw.iloc[0:0].copy()

            empresas_80_table = plantas_80[['empresa', 'market_share_aprox_pct', 'source_url']].copy()
            empresas_80_table['origen'] = 'plantas_80'
            empresas_80_table = empresas_80_table.rename(
                columns={'market_share_aprox_pct': 'share_aprox_pct', 'source_url': 'url_source'}
            )

            empresas_20_table = plantas_20[['empresa', 'export_share_aprox_pct', 'source_url']].copy()
            empresas_20_table['origen'] = 'exportadores_20'
            empresas_20_table = empresas_20_table.rename(
                columns={'export_share_aprox_pct': 'share_aprox_pct', 'source_url': 'url_source'}
            )
            em_company_share = (
                em_filtered
                .groupby(['empresa', 'sector_especialidad', 'url_source'], as_index=False)['Trade Value']
                .sum()
            )
            em_sector_totals = em_filtered.groupby('sector_especialidad', as_index=False)['Trade Value'].sum().rename(
                columns={'Trade Value': 'sector_total'}
            )
            em_company_share = em_company_share.merge(em_sector_totals, on='sector_especialidad', how='left')
            em_company_share['share_aprox_pct'] = np.where(
                em_company_share['sector_total'] > 0,
                (em_company_share['Trade Value'] / em_company_share['sector_total']) * 100,
                np.nan
            )
            em_company_table = em_company_share[['empresa', 'share_aprox_pct', 'url_source']].copy()
            em_company_table['origen'] = 'em_int'

            empresas_share_table = pd.concat([empresas_80_table, empresas_20_table, em_company_table], ignore_index=True)
            empresas_share_table = empresas_share_table.dropna(subset=['empresa']).copy()
            empresas_share_table['share_aprox_pct'] = pd.to_numeric(empresas_share_table['share_aprox_pct'], errors='coerce')
            empresas_share_table = (
                empresas_share_table
                .sort_values(['empresa', 'share_aprox_pct'], ascending=[True, False])
                .groupby('empresa', as_index=False)
                .agg({
                    'share_aprox_pct': 'max',
                    'url_source': 'first'
                })
                .sort_values('share_aprox_pct', ascending=False)
            )

            # Ajusta shares de empresas de equipo médico con el vector objetivo (normalizado a 100%).
            medtech_share_raw = {
                'Medtronic': 11.0,
                'GE HealthCare': 9.0,
                'Siemens Healthineers': 8.5,
                'Philips Healthcare': 7.0,
                'Abbott': 8.0,
                'Johnson & Johnson MedTech': 6.5,
                'Stryker': 5.5,
                'Becton Dickinson': 5.5,
                'Boston Scientific': 4.5,
                '3M Health Care': 3.5,
                'Cardinal Health': 3.5,
                'Thermo Fisher Scientific': 4.5,
                'Fresenius Medical Care': 3.5,
                'Zimmer Biomet': 3.5,
                'Baxter International': 3.5,
                'Hologic': 2.5,
                'Intuitive Surgical': 2.5,
                'Danaher': 2.5,
                'ResMed': 1.5,
                'Olympus Medical': 2.5
            }
            medtech_total = sum(medtech_share_raw.values())
            medtech_share_norm = {
                key: value * (100.0 / medtech_total)
                for key, value in medtech_share_raw.items()
            }

            def normalize_company_name(name: str) -> str:
                text = str(name).lower()
                text = text.replace("&", " and ")
                text = re.sub(r"[^a-z0-9]+", " ", text)
                return " ".join(text.split())

            medtech_share_lookup = {
                normalize_company_name(company): share
                for company, share in medtech_share_norm.items()
            }
            empresas_share_table['empresa_key'] = empresas_share_table['empresa'].map(normalize_company_name)
            mapped_share = empresas_share_table['empresa_key'].map(medtech_share_lookup)
            empresas_share_table['share_aprox_pct'] = np.where(
                mapped_share.notna(),
                mapped_share,
                empresas_share_table['share_aprox_pct']
            )
            empresas_share_table = empresas_share_table.drop(columns=['empresa_key'])

            empresas_share_table['share_aprox_pct'] = empresas_share_table['share_aprox_pct'].map(
                lambda x: f"{x:.1f}%" if pd.notna(x) else ""
            )

            plantas_80 = plantas_80[['empresa', 'planta_ciudad_aprox', 'estado', 'latitud_aprox', 'longitud_aprox']].copy()
            plantas_80['grupo'] = 'plantas_80'
            plantas_20 = plantas_20[['empresa', 'planta_ciudad_aprox', 'estado', 'latitud_aprox', 'longitud_aprox']].copy()
            plantas_20['grupo'] = 'plantas_20'

            plantas_map = pd.concat([plantas_20, plantas_80], ignore_index=True)
            em_map = em_filtered[['empresa', 'latitud_planta_mexico', 'longitud_planta_mexico', 'sector_grupo']].copy()
            em_map = em_map.rename(
                columns={
                    'latitud_planta_mexico': 'latitud_aprox',
                    'longitud_planta_mexico': 'longitud_aprox'
                }
            )
            em_map['planta_ciudad_aprox'] = 'Ubicación por coordenadas'
            em_map['estado'] = ''
            em_map['grupo'] = 'em_int'
            em_map = em_map[['empresa', 'planta_ciudad_aprox', 'estado', 'latitud_aprox', 'longitud_aprox', 'grupo']]
            plantas_map = pd.concat([plantas_map, em_map], ignore_index=True)
            plantas_map = plantas_map.dropna(subset=['latitud_aprox', 'longitud_aprox'])
            if not plantas_map.empty:
                coord_regions = (
                    plantas_map[['latitud_aprox', 'longitud_aprox']]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                coord_regions = assign_mexico_regions(coord_regions)
                plantas_map = plantas_map.merge(
                    coord_regions,
                    on=['latitud_aprox', 'longitud_aprox'],
                    how='left'
                )
                if selected_region_cluster != "Todo":
                    plantas_map = plantas_map[
                        plantas_map['region_cluster'] == selected_region_cluster
                    ].copy()
                    empresas_region_activas = set(
                        plantas_map['empresa'].dropna().astype(str).unique().tolist()
                    )
                    empresas_share_table = empresas_share_table[
                        empresas_share_table['empresa'].astype(str).isin(empresas_region_activas)
                    ].copy()

            if not plantas_map.empty:
                bubble_map = (
                    plantas_map
                    .groupby(['latitud_aprox', 'longitud_aprox', 'region_cluster'], as_index=False)
                    .agg(
                        total_plantas=('empresa', 'count'),
                        empresas_unicas=('empresa', 'nunique'),
                        empresas_lista=('empresa', lambda s: "<br>".join(sorted(pd.Series(s).dropna().astype(str).unique())))
                    )
                    .sort_values('total_plantas', ascending=False)
                )
                bubble_map['ubicacion'] = bubble_map['empresas_lista']

                fig_mexico_map = px.scatter_geo(
                    bubble_map,
                    lat='latitud_aprox',
                    lon='longitud_aprox',
                    size='total_plantas',
                    color='total_plantas',
                    hover_name='ubicacion',
                    hover_data={
                        'total_plantas': True,
                        'empresas_unicas': True,
                        'region_cluster': True,
                        'empresas_lista': False,
                        'latitud_aprox': ':.4f',
                        'longitud_aprox': ':.4f'
                    },
                    title='Bubble Map de Plantas en México (plantas_20 + plantas_80)',
                    color_continuous_scale='Turbo',
                    size_max=40
                )
                fig_mexico_map.update_geos(
                    scope='north america',
                    resolution=50,
                    lataxis_range=[14, 33],
                    lonaxis_range=[-119, -86],
                    showcountries=True,
                    countrycolor='#0f172a',
                    countrywidth=1.6,
                    showsubunits=True,
                    subunitcolor='#1f2937',
                    subunitwidth=1.4,
                    showland=True,
                    landcolor='#fef3c7',
                    showocean=True,
                    oceancolor='#dbeafe',
                    showcoastlines=True,
                    coastlinecolor='#0f172a',
                    coastlinewidth=1.0
                )
                fig_mexico_map.update_traces(
                    marker=dict(opacity=0.85, line=dict(width=1.1, color='#111827'))
                )
                fig_mexico_map.update_layout(
                    height=560,
                    margin=dict(l=10, r=10, t=60, b=10),
                    coloraxis_colorbar=dict(title='Plantas')
                )
                st.plotly_chart(fig_mexico_map, use_container_width=True)

            col_tab_1, col_tab_2 = st.columns([1.4, 1.1], gap="small")
            with col_tab_1:
                left_cols = ['Year', 'Month', 'HS2 4 Digit', 'HS4 4 Digit', 'Trade Value', 'Share']
                left_sources = []
                if not df_exports_filtered.empty:
                    left_sources.append(df_exports_filtered[left_cols].copy())
                if not em_filtered.empty:
                    left_sources.append(em_filtered[left_cols].copy())

                if left_sources:
                    df_left_table = (
                        pd.concat(left_sources, ignore_index=True)
                        .drop_duplicates(subset=left_cols)
                        .sort_values(['Year', 'Month', 'HS2 4 Digit', 'HS4 4 Digit'])
                    )
                else:
                    df_left_table = pd.DataFrame(columns=left_cols)

                st.dataframe(
                    df_left_table,
                    use_container_width=True,
                    hide_index=True
                )
            with col_tab_2:
                st.dataframe(
                    empresas_share_table[['empresa', 'share_aprox_pct', 'url_source']],
                    use_container_width=True,
                    hide_index=True
                )
    except Exception as e:
        st.error(f"Error al cargar exportaciones: {e}")

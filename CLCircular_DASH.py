# Librerías
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import streamlit as st
import unicodedata
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


def normalize_state_name(name: str) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = " ".join(text.split())
    aliases = {
        "coahuila de zaragoza": "coahuila",
        "michoacan de ocampo": "michoacan",
        "veracruz de ignacio de la llave": "veracruz",
        "mexico": "estado de mexico",
        "ciudad de mexico": "ciudad de mexico"
    }
    return aliases.get(text, text)


def resolve_data_file(base_dir: Path, filename: str):
    candidates = [
        base_dir / filename,
        base_dir / 'Code' / filename,
        base_dir.parent / filename,
        base_dir.parent / 'Code' / filename,
        Path.cwd() / filename,
        Path.cwd() / 'Code' / filename
    ]
    return next((p for p in candidates if p.exists()), None)


@st.cache_data(show_spinner=False)
def load_origin_exports(base_dir_str: str) -> pd.DataFrame:
    base_dir = Path(base_dir_str)
    direct_candidates = [
        base_dir / "Origen exports",
        base_dir / "Origen Exports",
        base_dir / "Code" / "Origen exports",
        base_dir / "Code" / "Origen Exports",
        base_dir.parent / "Origen exports",
        base_dir.parent / "Origen Exports",
        base_dir.parent / "Code" / "Origen exports",
        base_dir.parent / "Code" / "Origen Exports",
        Path.cwd() / "Origen exports",
        Path.cwd() / "Origen Exports",
        Path.cwd() / "Code" / "Origen exports",
        Path.cwd() / "Code" / "Origen Exports",
    ]

    origin_dirs = [p for p in direct_candidates if p.exists() and p.is_dir()]

    if not origin_dirs:
        # Fallback robusto para despliegues donde cambia el cwd/base_dir.
        search_roots = [base_dir, base_dir.parent, Path.cwd()]
        seen = set()
        for root in search_roots:
            if root in seen or not root.exists():
                continue
            seen.add(root)
            for p in root.rglob("*"):
                if p.is_dir() and p.name.strip().lower() == "origen exports":
                    origin_dirs.append(p)
        # Evita duplicados conservando orden.
        origin_dirs = list(dict.fromkeys(origin_dirs))

    if not origin_dirs:
        return pd.DataFrame()

    files = []
    for origin_dir in origin_dirs:
        files.extend(sorted(list(origin_dir.glob("*.xlsx")) + list(origin_dir.glob("*.csv"))))
    files = list(dict.fromkeys(files))
    if not files:
        return pd.DataFrame()

    dfs = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".xlsx":
                dfs.append(pd.read_excel(file_path))
            else:
                try:
                    dfs.append(pd.read_csv(file_path))
                except UnicodeDecodeError:
                    dfs.append(pd.read_csv(file_path, encoding="latin-1"))
        except Exception:
            # Si un archivo falla, continúa con el resto para no romper el dashboard.
            continue

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_df_estado_heatmap(base_dir_str: str) -> pd.DataFrame:
    base_dir = Path(base_dir_str)
    delitos_dir_candidates = [
        base_dir.parent / 'Municipal-Delitos-2015-2025_ene2026',
        base_dir / 'Municipal-Delitos-2015-2025_ene2026',
        base_dir.parent.parent / 'Municipal-Delitos-2015-2025_ene2026'
    ]
    delitos_dir = next((p for p in delitos_dir_candidates if p.exists() and p.is_dir()), None)
    if delitos_dir is None:
        return pd.DataFrame(columns=['Entidad', 'estado_key', 'Promedio_anual_delitos'])

    file_names = [
        '2015.xlsx', '2016.xlsx', '2017.xlsx', '2018.xlsx', '2019.xlsx',
        '2020.xlsx', '2021.xlsx', '2022.xlsx', '2023.xlsx',
        '2024_ene2026.xlsx', '2025_ene2026.xlsx'
    ]
    files = [delitos_dir / name for name in file_names if (delitos_dir / name).exists()]
    if not files:
        return pd.DataFrame(columns=['Entidad', 'Promedio_anual_delitos'])

    dfs = []
    for file_path in files:
        df_file = pd.read_excel(file_path)
        df_file['source_file'] = str(file_path)
        dfs.append(df_file)
    combined = pd.concat(dfs, ignore_index=True)

    tipo_s = combined['Tipo de delito'].astype(str).str.strip().str.casefold()
    subtipo_s = combined['Subtipo de delito'].astype(str).str.strip().str.casefold()
    modalidad_s = combined['Modalidad'].astype(str).str.strip().str.casefold()

    tipos_objetivo = [
        "Homicidio", "Lesiones", "Feminicidio", "Secuestro", "Narcomenudeo", "Trata de personas"
    ]
    tipos_objetivo_s = [t.casefold() for t in tipos_objetivo]
    subtipos_directos_s = [s.casefold() for s in ["Extorsión", "Fraude", "Daño a la propiedad"]]

    mask_tipos = tipo_s.isin(tipos_objetivo_s)
    mask_robo_negocio = (
        (subtipo_s == "robo a negocio".casefold()) &
        modalidad_s.isin(["con violencia".casefold(), "sin violencia".casefold()])
    )
    mask_subtipos_directos = subtipo_s.isin(subtipos_directos_s)
    mask_robo_maquinaria = (
        (subtipo_s == "robo de maquinaria".casefold()) &
        modalidad_s.isin([
            "robo de herramienta industrial o agricola con violencia".casefold(),
            "robo de herramienta industrial o agricola sin violencia".casefold()
        ])
    )
    mask_robo_transportista = (
        (subtipo_s == "robo a transportista".casefold()) &
        modalidad_s.isin(["con violencia".casefold(), "sin violencia".casefold()])
    )

    df_delitos_objetivo = combined[mask_tipos | mask_robo_negocio | mask_subtipos_directos | mask_robo_maquinaria | mask_robo_transportista].copy()

    meses = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    df_delitos_objetivo[meses] = df_delitos_objetivo[meses].apply(pd.to_numeric, errors='coerce').fillna(0)
    df_delitos_objetivo['Total_anual_registro'] = df_delitos_objetivo[meses].sum(axis=1)

    estado_anio = (
        df_delitos_objetivo
        .groupby(['Entidad', 'Año'], as_index=False)['Total_anual_registro']
        .sum()
    )
    df_estado = (
        estado_anio
        .groupby('Entidad', as_index=False)['Total_anual_registro']
        .mean()
        .rename(columns={'Total_anual_registro': 'Promedio_anual_delitos'})
    )
    df_estado['estado_key'] = df_estado['Entidad'].map(normalize_state_name)
    return df_estado[['Entidad', 'estado_key', 'Promedio_anual_delitos']]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_distance_km(lat, lon, points_df):
    if points_df.empty:
        return np.inf
    dists = haversine_km(lat, lon, points_df['lat'].values, points_df['lon'].values)
    return float(np.nanmin(dists))


def point_in_polygon(lat, lon, polygon_latlon):
    # Ray-casting algorithm for point-in-polygon using (lat, lon) tuples.
    inside = False
    n = len(polygon_latlon)
    if n < 3:
        return False
    p1_lat, p1_lon = polygon_latlon[0]
    for i in range(n + 1):
        p2_lat, p2_lon = polygon_latlon[i % n]
        if lon > min(p1_lon, p2_lon):
            if lon <= max(p1_lon, p2_lon):
                if lat <= max(p1_lat, p2_lat):
                    if p1_lon != p2_lon:
                        lat_intersect = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                    else:
                        lat_intersect = p1_lat
                    if p1_lat == p2_lat or lat <= lat_intersect:
                        inside = not inside
        p1_lat, p1_lon = p2_lat, p2_lon
    return inside


MEXICO_POLYGON_LATLON = [
    (32.7, -117.2), (32.6, -114.7), (31.3, -111.1), (31.3, -109.0), (31.8, -108.2),
    (31.8, -106.5), (30.7, -105.0), (29.7, -104.0), (29.0, -103.2), (27.8, -101.8),
    (26.7, -100.4), (25.6, -99.0), (24.8, -97.7), (23.8, -96.6), (22.8, -95.5),
    (21.6, -94.5), (20.3, -93.4), (18.8, -92.2), (17.9, -91.3), (18.3, -90.6),
    (20.0, -89.7), (21.4, -88.5), (21.6, -87.2), (20.5, -86.8), (19.5, -87.3),
    (18.3, -89.5), (17.0, -91.6), (16.0, -93.0), (15.2, -95.0), (15.0, -97.0),
    (15.6, -99.2), (16.5, -101.5), (17.8, -103.8), (19.2, -105.6), (21.0, -106.9),
    (23.0, -108.3), (24.8, -109.3), (26.5, -110.1), (28.0, -111.3), (29.2, -112.4),
    (30.3, -113.6), (31.2, -114.8), (32.2, -116.0), (32.7, -117.2)
]


def compute_strategic_point_cluster(farma_geo, maquinaria_geo, hubs_geo, dist_geo, risk_points):
    if farma_geo.empty and maquinaria_geo.empty and hubs_geo.empty and dist_geo.empty:
        return None

    farma_center = None
    maquinaria_center = None
    hubs_center = None
    dist_center = None

    if not farma_geo.empty:
        farma_points = farma_geo[['lat', 'lon']].to_numpy(dtype=float)
        _, farma_centroids = kmeans_2d(farma_points, k=min(4, len(farma_points)))
        if len(farma_centroids) > 0:
            farma_center = farma_centroids.mean(axis=0)

    if not maquinaria_geo.empty:
        maq_points = maquinaria_geo[['lat', 'lon']].to_numpy(dtype=float)
        _, maq_centroids = kmeans_2d(maq_points, k=min(4, len(maq_points)))
        if len(maq_centroids) > 0:
            maquinaria_center = maq_centroids.mean(axis=0)

    if not hubs_geo.empty:
        hubs_points = hubs_geo[['lat', 'lon']].to_numpy(dtype=float)
        _, hubs_centroids = kmeans_2d(hubs_points, k=min(3, len(hubs_points)))
        if len(hubs_centroids) > 0:
            hubs_center = hubs_centroids.mean(axis=0)

    if not dist_geo.empty:
        dist_points = dist_geo[['lat', 'lon']].to_numpy(dtype=float)
        _, dist_centroids = kmeans_2d(dist_points, k=min(3, len(dist_points)))
        if len(dist_centroids) > 0:
            dist_center = dist_centroids.mean(axis=0)

    centers = [c for c in [farma_center, maquinaria_center, hubs_center, dist_center] if c is not None]
    if not centers:
        return None
    centers_arr = np.vstack(centers)
    target_lat = float(centers_arr[:, 0].mean())
    target_lon = float(centers_arr[:, 1].mean())

    lat_grid = np.linspace(14.5, 32.5, 70)
    lon_grid = np.linspace(-117.0, -86.0, 90)
    candidates = pd.DataFrame([(la, lo) for la in lat_grid for lo in lon_grid], columns=['lat', 'lon'])
    candidates = candidates[
        candidates.apply(
            lambda r: point_in_polygon(float(r['lat']), float(r['lon']), MEXICO_POLYGON_LATLON),
            axis=1
        )
    ].reset_index(drop=True)
    if candidates.empty:
        return None

    def risk_penalty(lat, lon):
        if risk_points.empty:
            return 0.0
        d_risk = haversine_km(lat, lon, risk_points['lat'].values, risk_points['lon'].values)
        return float(np.sum(risk_points['risk_weighted'].values / (d_risk + 25.0)))

    # Objective: prioritize minimum distance to visible layers; risk as light penalty.
    scores = []
    for _, row in candidates.iterrows():
        lat = float(row['lat'])
        lon = float(row['lon'])
        d_target = haversine_km(lat, lon, target_lat, target_lon)
        d_farma_near = nearest_distance_km(lat, lon, farma_geo)
        d_maq_near = nearest_distance_km(lat, lon, maquinaria_geo)
        d_hubs_near = nearest_distance_km(lat, lon, hubs_geo)
        d_dist_near = nearest_distance_km(lat, lon, dist_geo)

        layer_terms = []
        if not farma_geo.empty:
            layer_terms.append(("farma", d_farma_near))
        if not maquinaria_geo.empty:
            layer_terms.append(("maquinaria", d_maq_near))
        if not hubs_geo.empty:
            layer_terms.append(("hubs", d_hubs_near))
        if not dist_geo.empty:
            layer_terms.append(("dist", d_dist_near))

        layer_weight_map = {
            "farma": 0.35,
            "maquinaria": 0.30,
            "hubs": 0.20,
            "dist": 0.15
        }
        w_sum = sum(layer_weight_map[k] for k, _ in layer_terms)
        distance_component = 0.0
        if w_sum > 0:
            for key, dist_km in layer_terms:
                distance_component += (layer_weight_map[key] / w_sum) * dist_km

        penalty = risk_penalty(lat, lon)
        objective = (0.75 * distance_component) + (0.25 * d_target) + (0.08 * penalty)
        scores.append(objective)

    candidates['objective'] = scores
    best = candidates.loc[candidates['objective'].idxmin()]
    return {'lat': float(best['lat']), 'lon': float(best['lon']), 'score': float(best['objective'])}


def montecarlo_risk_forecast(strategic_point, farma_geo, hubs_geo, dist_geo, risk_points, n_simulaciones=1000):
    if strategic_point is None:
        return None

    lat = strategic_point['lat']
    lon = strategic_point['lon']

    d_farma = nearest_distance_km(lat, lon, farma_geo)
    d_hubs = nearest_distance_km(lat, lon, hubs_geo)
    d_dist = nearest_distance_km(lat, lon, dist_geo)

    if risk_points.empty:
        risk_local = 0.0
    else:
        d_risk = haversine_km(lat, lon, risk_points['lat'].values, risk_points['lon'].values)
        risk_local = float(np.sum(risk_points['risk_weighted'].values / (d_risk + 25.0)))

    def clip_0_10(x):
        return float(np.clip(x, 0.0, 10.0))

    # Scores base (0-10)
    demanda = clip_0_10(10.0 - (0.030 * d_farma + 0.015 * d_dist))
    seguridad = clip_0_10(10.0 - (1.8 * risk_local))
    operacion = clip_0_10(10.0 - (0.022 * d_farma + 0.018 * d_hubs + 0.015 * d_dist))
    financiero = clip_0_10(0.45 * operacion + 0.35 * seguridad + 0.20 * demanda)

    sedes = {
        'Punto Estratégico': {
            'Demanda': demanda,
            'Seguridad': seguridad,
            'Operacion': operacion,
            'Financiero': financiero
        }
    }

    resultados_finales = {}
    for ciudad, valores in sedes.items():
        calificaciones = []
        for _ in range(n_simulaciones):
            score = (
                np.random.normal(valores['Demanda'], 0.5) +
                np.random.normal(valores['Seguridad'], 0.8) +
                np.random.normal(valores['Operacion'], 0.6) +
                np.random.normal(valores['Financiero'], 0.4)
            ) / 4
            calificaciones.append(score)

        calificaciones = np.array(calificaciones)
        resultados_finales[ciudad] = {
            'Calificación Promedio': float(np.mean(calificaciones)),
            'Riesgo (Desviación)': float(np.std(calificaciones)),
            'P95 (Escenario Pesimista)': float(np.percentile(calificaciones, 5)),
            'Demanda base': demanda,
            'Seguridad base': seguridad,
            'Operación base': operacion,
            'Financiero base': financiero
        }

    return resultados_finales['Punto Estratégico']


### Title and Page configuration

st.set_page_config(
    page_title="CLCircular - Dashboard",
    page_icon = "https://play-lh.googleusercontent.com/CMVIb6hsKkX7-4wlAZMfVOzVFbg6zCAI3MJFXtRbARXRhZbWHDLAKNSNuCOuM1i1gQ=w240-h480-rw",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 10% 5%, #f6f9fc 0%, #eef3f9 35%, #f7f9fc 100%);
    }
    .block-container {
        padding-top: 2.8rem;
        padding-bottom: 1.2rem;
    }
    .dashboard-hero {
        background: #00296b;
        border: 1px solid #00296b;
        border-radius: 14px;
        padding: 1rem 1.1rem 0.8rem 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 10px 28px rgba(0, 41, 107, 0.22);
    }
    .dashboard-header {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .dashboard-logo {
        width: 300px;
        height: 300px;
        object-fit: contain;
        border-radius: 10px;
        margin-top: -16px;
    }
    .dashboard-hero-wrap {
        flex: 1;
    }
    .dashboard-hero h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }
    .dashboard-hero p {
        margin: 0.25rem 0 0 0;
        color: #d1fae5;
        font-size: 1.08rem;
    }
    .dashboard-hero .hero-description {
        margin-top: 0.55rem;
        color: #ecfeff;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        border: 1px solid #dbe4ee;
        color: #0f172a;
        padding: 0.45rem 0.75rem;
    }
    .stTabs [aria-selected="true"] {
        background: #94bf43 !important;
        color: #ffffff !important;
        border-color: #94bf43 !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eff6ff 100%);
        border-right: 1px solid #dbeafe;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e9ba2c;
        border-radius: 12px;
        padding: 0.55rem 0.8rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #0196b9;
        border-radius: 16px;
        padding: 0.75rem 0.75rem 0.35rem 0.75rem;
        margin-top: 0.45rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stPlotlyChart"] > div {
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stToggle"] {
        margin-bottom: 0.35rem;
    }
    [data-testid="stToggle"] label {
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 0.2rem !important;
    }
    [data-testid="stToggle"] p {
        font-size: 1.02rem !important;
        font-weight: 600 !important;
        margin: 0.1rem 0 0 0 !important;
        order: 2 !important;
    }
    [data-testid="stToggle"] div[role="switch"] {
        transform: scale(1.2);
        transform-origin: left center;
        order: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="dashboard-header">
      <img class="dashboard-logo" src="https://play-lh.googleusercontent.com/CMVIb6hsKkX7-4wlAZMfVOzVFbg6zCAI3MJFXtRbARXRhZbWHDLAKNSNuCOuM1i1gQ=w240-h480-rw" alt="logo">
      <div class="dashboard-hero-wrap">
        <div class="dashboard-hero">
          <h1>CLCircular Strategic Intelligence Hub</h1>
          <p>Pharma Industry Intelligence Platform</p>
          <div class="hero-description">
            Este dashboard proporciona una plataforma de inteligencia estratégica para la expansión de CLCircular en el mercado farmacéutico mexicano, integrando análisis de mercado, evaluación de alianzas con hubs logísticos, clústeres industriales y empresas de transporte, así como la identificación de ubicaciones óptimas para establecer un hub estratégico propio. El análisis se apoya en modelos SARIMA para proyecciones de mercado, segmentación mediante K-means y simulaciones Monte Carlo, permitiendo identificar oportunidades, riesgos y corredores logísticos clave dentro del ecosistema farmacéutico en México.
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

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
                df_usa_filtered = df_usa_year_filtered[df_usa_year_filtered['Category'].isin(categories)].sort_values(['Category', 'Year'])

                if df_usa_filtered.empty:
                    st.warning("Selecciona al menos una categoría para mostrar la gráfica.")
                else:
                    yearly_totals = (
                        df_usa_filtered
                        .groupby('Year', as_index=False)['Value']
                        .sum()
                        .sort_values('Year')
                    )
                    current_market_value = float(yearly_totals.iloc[-1]['Value'])
                    first_year = int(yearly_totals.iloc[0]['Year'])
                    last_year = int(yearly_totals.iloc[-1]['Year'])
                    first_value = float(yearly_totals.iloc[0]['Value'])
                    last_value = float(yearly_totals.iloc[-1]['Value'])

                    growth_display = "N/A"
                    avg_growth_display = "N/A"
                    growth_money_display = "N/A"
                    avg_growth_money_display = "N/A"
                    if first_value != 0:
                        growth_pct = ((last_value - first_value) / first_value) * 100
                        growth_display = f"{growth_pct:,.1f}%"
                        growth_money_display = f"${(last_value - first_value):,.1f} M"
                    year_span = max(last_year - first_year, 0)
                    if first_value > 0 and last_value > 0 and year_span > 0:
                        avg_growth_pct = ((last_value / first_value) ** (1 / year_span) - 1) * 100
                        avg_growth_display = f"{avg_growth_pct:,.1f}%"
                        avg_growth_money_display = f"${((last_value - first_value) / year_span):,.1f} M/año"

                    kpi_ctx_1, kpi_ctx_2, kpi_ctx_3 = st.columns(3)
                    with kpi_ctx_1:
                        st.metric("Valor total mercado actual", f"${current_market_value:,.1f} M")
                    with kpi_ctx_2:
                        st.metric(
                            f"Crecimiento % ({first_year}-{last_year})",
                            growth_display
                        )
                        st.caption(f"Equivalente en dinero: {growth_money_display}")
                    with kpi_ctx_3:
                        st.metric("Crecimiento promedio anual", avg_growth_display)
                        st.caption(f"Equivalente en dinero: {avg_growth_money_display}")

                    category_filter = st.multiselect(
                        "Filtra categorías",
                        options=categories,
                        default=categories
                    )
                    df_usa_filtered = df_usa_year_filtered[
                        df_usa_year_filtered['Category'].isin(category_filter)
                    ].sort_values(['Category', 'Year'])
                    if df_usa_filtered.empty:
                        st.warning("Selecciona al menos una categoría para mostrar la gráfica.")
                    else:
                        fig = px.line(
                            df_usa_filtered,
                            x='Year',
                            y='Value',
                            color='Category',
                            title='Tamaño de la industria Farmacéutica en U.S.A.',
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
                kpi_sarima_placeholder = st.container()
                sarima_filter = st.multiselect(
                    "Filtra categorías SARIMA",
                    options=sarima_categories,
                    default=sarima_categories,
                    key="sarima_filter_ctx"
                )
                forecast_steps = st.slider(
                    "Años de pronóstico (SARIMA)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="forecast_steps_ctx"
                )
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
                    with kpi_sarima_placeholder:
                        current_market_value = float(
                            sum(observed_df['Value'].iloc[-1] for _, observed_df, _, _ in series_data if not observed_df.empty)
                        )
                        forecast_market_value = float(
                            sum(forecast_df['Value'].iloc[-1] for _, _, forecast_df, _ in series_data if not forecast_df.empty)
                        )

                        growth_x_display = "N/A"
                        annual_growth_display = "N/A"
                        growth_x_money_display = "N/A"
                        annual_growth_money_display = "N/A"

                        if current_market_value != 0:
                            growth_x_pct = ((forecast_market_value - current_market_value) / current_market_value) * 100
                            growth_x_display = f"{growth_x_pct:,.1f}%"
                            growth_x_money_display = f"${(forecast_market_value - current_market_value):,.1f} M"
                        if current_market_value > 0 and forecast_market_value > 0 and forecast_steps > 0:
                            annual_growth_pct = ((forecast_market_value / current_market_value) ** (1 / forecast_steps) - 1) * 100
                            annual_growth_display = f"{annual_growth_pct:,.1f}%"
                            annual_growth_money_display = f"${((forecast_market_value - current_market_value) / forecast_steps):,.1f} M/año"

                        kpi_sar_1, kpi_sar_2, kpi_sar_3 = st.columns(3)
                        with kpi_sar_1:
                            st.metric(
                                f"Valor de mercado en {forecast_steps} años",
                                f"${forecast_market_value:,.1f} M"
                            )
                        with kpi_sar_2:
                            st.metric(
                                f"Crecimiento porcentual del mercado en {forecast_steps} años",
                                growth_x_display
                            )
                            st.caption(f"Equivalente en dinero: {growth_x_money_display}")
                        with kpi_sar_3:
                            st.metric("Crecimiento porcentual anual", annual_growth_display)
                            st.caption(f"Equivalente en dinero: {annual_growth_money_display}")

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
        subsegment_kpi_placeholder = st.container()

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
            df_subsegments_yearly = (
                df_subsegments_trend
                .groupby('Year', as_index=False)['Value']
                .sum()
                .sort_values('Year')
            )

            if view_mode == "Gráfica normal":
                current_subsegment_value = float(df_subsegments_yearly.iloc[-1]['Value'])
                first_subsegment_year = int(df_subsegments_yearly.iloc[0]['Year'])
                last_subsegment_year = int(df_subsegments_yearly.iloc[-1]['Year'])
                first_subsegment_value = float(df_subsegments_yearly.iloc[0]['Value'])
                last_subsegment_value = float(df_subsegments_yearly.iloc[-1]['Value'])

                sub_growth_display = "N/A"
                sub_growth_money_display = "N/A"
                sub_annual_growth_display = "N/A"
                sub_annual_growth_money_display = "N/A"
                if first_subsegment_value != 0:
                    sub_growth = ((last_subsegment_value - first_subsegment_value) / first_subsegment_value) * 100
                    sub_growth_display = f"{sub_growth:,.1f}%"
                    sub_growth_money_display = f"${(last_subsegment_value - first_subsegment_value):,.1f} M"
                sub_year_span = max(last_subsegment_year - first_subsegment_year, 0)
                if first_subsegment_value > 0 and last_subsegment_value > 0 and sub_year_span > 0:
                    sub_annual_growth = ((last_subsegment_value / first_subsegment_value) ** (1 / sub_year_span) - 1) * 100
                    sub_annual_growth_display = f"{sub_annual_growth:,.1f}%"
                    sub_annual_growth_money_display = f"${((last_subsegment_value - first_subsegment_value) / sub_year_span):,.1f} M/año"

                with subsegment_kpi_placeholder:
                    kpi_sub_1, kpi_sub_2, kpi_sub_3 = st.columns(3)
                    with kpi_sub_1:
                        st.metric("Valor total subsegmento actual", f"${current_subsegment_value:,.1f} M")
                    with kpi_sub_2:
                        st.metric(
                            f"Crecimiento % ({first_subsegment_year}-{last_subsegment_year})",
                            sub_growth_display
                        )
                        st.caption(f"Equivalente en dinero: {sub_growth_money_display}")
                    with kpi_sub_3:
                        st.metric("Crecimiento promedio anual", sub_annual_growth_display)
                        st.caption(f"Equivalente en dinero: {sub_annual_growth_money_display}")

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
                sarima_subsegment_totals = []

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
                        current_value_cat = float(cat_df['Value'].iloc[-1])
                        forecast_value_cat = float(forecast.values[-1])
                        sarima_subsegment_totals.append((current_value_cat, forecast_value_cat))
                        modeled_count += 1
                    except Exception as sarima_error:
                        st.warning(f"No se pudo ajustar SARIMA para {category}: {sarima_error}")

                if modeled_count == 0:
                    st.error("No se pudo generar la evolución SARIMA de subsegmentos con la selección actual.")
                else:
                    current_subsegment_value = float(sum(x[0] for x in sarima_subsegment_totals))
                    forecast_subsegment_value = float(sum(x[1] for x in sarima_subsegment_totals))
                    sub_growth_display = "N/A"
                    sub_growth_money_display = "N/A"
                    sub_annual_growth_display = "N/A"
                    sub_annual_growth_money_display = "N/A"
                    if current_subsegment_value != 0:
                        sub_growth = ((forecast_subsegment_value - current_subsegment_value) / current_subsegment_value) * 100
                        sub_growth_display = f"{sub_growth:,.1f}%"
                        sub_growth_money_display = f"${(forecast_subsegment_value - current_subsegment_value):,.1f} M"
                    if current_subsegment_value > 0 and forecast_subsegment_value > 0 and subsegment_forecast_steps > 0:
                        sub_annual_growth = ((forecast_subsegment_value / current_subsegment_value) ** (1 / subsegment_forecast_steps) - 1) * 100
                        sub_annual_growth_display = f"{sub_annual_growth:,.1f}%"
                        sub_annual_growth_money_display = f"${((forecast_subsegment_value - current_subsegment_value) / subsegment_forecast_steps):,.1f} M/año"

                    with subsegment_kpi_placeholder:
                        kpi_sub_1, kpi_sub_2, kpi_sub_3 = st.columns(3)
                        with kpi_sub_1:
                            st.metric(
                                f"Valor total subsegmento en {subsegment_forecast_steps} años",
                                f"${forecast_subsegment_value:,.1f} M"
                            )
                        with kpi_sub_2:
                            st.metric(
                                f"Crecimiento % en {subsegment_forecast_steps} años",
                                sub_growth_display
                            )
                            st.caption(f"Equivalente en dinero: {sub_growth_money_display}")
                        with kpi_sub_3:
                            st.metric("Crecimiento promedio anual", sub_annual_growth_display)
                            st.caption(f"Equivalente en dinero: {sub_annual_growth_money_display}")

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

        st.success("Datos cargados correctamente")

    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")

with tab_exporta:
    st.subheader("¿Qué se exporta?")
    try:
        base_dir = Path(__file__).resolve().parent
        us_imports_path = resolve_data_file(base_dir, 'exportaciones_farmaceuticas_identificadas.xlsx')
        em_int_path = resolve_data_file(base_dir, 'exportaciones_equipo_medico_empresas_sectores_urls.xlsx')
        plantas_path = resolve_data_file(base_dir, 'empresas_con_plantas_en_mexico_y_exportadores_MX_USA.xlsx')
        if us_imports_path is None or em_int_path is None or plantas_path is None:
            missing = []
            if us_imports_path is None:
                missing.append('exportaciones_farmaceuticas_identificadas.xlsx')
            if em_int_path is None:
                missing.append('exportaciones_equipo_medico_empresas_sectores_urls.xlsx')
            if plantas_path is None:
                missing.append('empresas_con_plantas_en_mexico_y_exportadores_MX_USA.xlsx')
            raise FileNotFoundError("Archivos faltantes para ¿Qué se exporta?: " + ", ".join(missing))

        us_imports = pd.read_excel(us_imports_path)
        em_int = pd.read_excel(em_int_path)
        df_origin_exports = load_origin_exports(str(base_dir))
        df_estado_heat = build_df_estado_heatmap(str(base_dir))
        hubs_candidates = [
            resolve_data_file(base_dir, 'hubs_y_organizaciones_adicionales_farmaceuticas_mexico_urls.xlsx'),
            resolve_data_file(base_dir, 'hubs_y_organizaciones_adicionales_farmaceuticas_mexico.xlsx')
        ]
        hubs_file = next((p for p in hubs_candidates if p is not None and p.exists()), None)
        df_hubs = pd.read_excel(hubs_file) if hubs_file is not None else pd.DataFrame()
        df_el_mexico = pd.DataFrame([
            {"empresa": "World Courier", "hub_mexico": "Tlalnepantla (Estado de México)", "latitud": 19.5407, "longitud": -99.1950},
            {"empresa": "Biocair", "hub_mexico": "Ciudad de México", "latitud": 19.4326, "longitud": -99.1332},
            {"empresa": "Marken", "hub_mexico": "Ciudad de México", "latitud": 19.4326, "longitud": -99.1332},
            {"empresa": "Cryoport", "hub_mexico": "Guadalajara", "latitud": 20.6597, "longitud": -103.3496},
            {"empresa": "Movianto", "hub_mexico": "Ciudad de México", "latitud": 19.4326, "longitud": -99.1332},
            {"empresa": "Bomi Group", "hub_mexico": "Querétaro", "latitud": 20.5888, "longitud": -100.3899},
            {"empresa": "FIEGE Pharma Logistics", "hub_mexico": "Monterrey", "latitud": 25.6866, "longitud": -100.3161},
            {"empresa": "JAS Worldwide", "hub_mexico": "Guadalajara", "latitud": 20.6597, "longitud": -103.3496},
            {"empresa": "GEODIS Pharma Healthcare", "hub_mexico": "Guadalajara", "latitud": 20.6597, "longitud": -103.3496},
            {"empresa": "Almac Group", "hub_mexico": "Ciudad de México", "latitud": 19.4326, "longitud": -99.1332}
        ])
        hs2_excluded = {
            'Preparaciones Alimenticias Diversas',
            'Instrumentos Musicales; sus Partes y Accesorios',
            'Jabones, Lubricantes, Ceras, Velas, Pastas de Modelar',
            'Aparatos de Relojería y sus Partes'
        }
        us_imports = us_imports[~us_imports['HS2 4 Digit'].astype(str).isin(hs2_excluded)].copy()
        em_int = em_int[~em_int['HS2 4 Digit'].astype(str).isin(hs2_excluded)].copy()
        plantas_80_raw = pd.read_excel(plantas_path, sheet_name='Producto_80', header=3)
        plantas_20_raw = pd.read_excel(plantas_path, sheet_name='Exportadores_20', header=3)

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

        region_states_map = {
            "Región Norte": [
                "Baja California", "Baja California Sur", "Sonora", "Chihuahua",
                "Coahuila", "Nuevo León", "Durango", "Sinaloa"
            ],
            "Región Sur": [
                "Chiapas", "Oaxaca", "Guerrero", "Puebla"
            ],
            "Región Golfo": [
                "Tamaulipas", "Veracruz", "Tabasco", "Campeche", "Yucatán", "Quintana Roo"
            ],
            "Región Centro": [
                "Aguascalientes", "Ciudad de México", "Estado de México", "Guanajuato",
                "Hidalgo", "Jalisco", "Michoacán", "Morelos", "Querétaro",
                "San Luis Potosí", "Tlaxcala", "Zacatecas", "Colima", "Nayarit"
            ]
        }
        region_state_keys_map = {
            region: {normalize_state_name(state) for state in states}
            for region, states in region_states_map.items()
        }

        state_catalog = pd.DataFrame(columns=['state_key', 'State_display'])
        if not df_origin_exports.empty and 'State' in df_origin_exports.columns:
            state_catalog = (
                df_origin_exports[['State']]
                .dropna()
                .assign(
                    State=lambda d: d['State'].astype(str).str.strip(),
                    state_key=lambda d: d['State'].map(normalize_state_name)
                )
            )
            state_catalog = state_catalog[state_catalog['state_key'].astype(str).str.len() > 0]
            state_catalog = (
                state_catalog
                .groupby('state_key', as_index=False)['State']
                .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
                .rename(columns={'State': 'State_display'})
            )

        available_state_keys = set(state_catalog['state_key'].tolist())
        if selected_region_cluster != "Todo":
            allowed_state_keys_region = available_state_keys.intersection(
                region_state_keys_map.get(selected_region_cluster, set())
            )
        else:
            allowed_state_keys_region = available_state_keys

        state_options = (
            state_catalog[state_catalog['state_key'].isin(allowed_state_keys_region)]
            .sort_values('State_display')['State_display']
            .tolist()
        )
        selected_states = st.sidebar.multiselect(
            "Estados",
            options=["Todos"] + state_options,
            default=["Todos"],
            key="selected_export_states"
        )
        if (not selected_states) or ("Todos" in selected_states):
            selected_states_effective = state_options
        else:
            selected_states_effective = selected_states
        selected_state_keys_effective = set(
            state_catalog[state_catalog['State_display'].isin(selected_states_effective)]['state_key'].tolist()
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
            df_exports_filtered['Trade Value Ajustado'] = (
                df_exports_filtered['Trade Value'] * df_exports_filtered['sector_multiplier']
            )
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
            avg_annual_growth_display = "N/A"
            avg_annual_growth_money_display = "N/A"
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
                years_span = max((last_period - first_period).days / 365.25, 0.0)
                if first_value > 0 and last_value > 0 and years_span > 0:
                    avg_annual_growth = ((last_value / first_value) ** (1 / years_span) - 1) * 100
                    avg_annual_growth_display = f"{avg_annual_growth:,.1f}%"
                    avg_annual_growth_money = (last_value - first_value) / years_span
                    avg_annual_growth_money_display = f"${avg_annual_growth_money:,.0f} / año"

            kpi_col_1, kpi_col_2, kpi_col_3 = st.columns(3)
            with kpi_col_1:
                st.metric("Valor total exportado (MXN)", f"${total_trade_value:,.0f}")
            with kpi_col_2:
                st.metric("Crecimiento % (inicio-fin)", growth_pct_display)
                if growth_range_label:
                    st.caption(growth_range_label)
            with kpi_col_3:
                st.metric("Crecimiento anual promedio", avg_annual_growth_display)
                st.caption(f"Crecimiento promedio: {avg_annual_growth_money_display}")

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
            hs2_short_map = {}
            hs2_used_short = {}
            for full_name in df_top_exports['HS2 4 Digit'].astype(str).tolist():
                base_short = full_name if len(full_name) <= 24 else full_name[:24].rstrip() + "..."
                count_short = hs2_used_short.get(base_short, 0) + 1
                hs2_used_short[base_short] = count_short
                short_name = base_short if count_short == 1 else f"{base_short} ({count_short})"
                hs2_short_map[full_name] = short_name
            df_top_exports['HS2 corto'] = df_top_exports['HS2 4 Digit'].map(hs2_short_map)
            df_top_exports_plot = df_top_exports.sort_values('Trade Value Ajustado', ascending=False).copy()
            fig_exports = px.bar(
                df_top_exports_plot,
                x='HS2 corto',
                y='Trade Value Ajustado',
                title=f'Top exportaciones ({selected_export_year_range[0]}-{selected_export_year_range[1]})',
                labels={'Trade Value Ajustado': 'Trade Value', 'HS2 corto': 'Producto HS2'},
                color='Trade Value Ajustado',
                color_continuous_scale=['#94bf43', '#7ab55f', '#5eb07b', '#3fa89a', '#0196b9'],
                custom_data=['HS4_hover', 'HS2 4 Digit']
            )
            fig_exports.update_traces(
                text=df_top_exports_plot['Trade Value Ajustado'].map(lambda x: f"${x:,.0f}"),
                textposition='outside',
                hovertemplate="<b>%{customdata[1]}</b><br>Trade Value: $%{y:,.0f}<br><br><b>HS4 4 Digit</b><br>%{customdata[0]}<extra></extra>"
            )
            fig_exports.update_yaxes(tickprefix="$", separatethousands=True)
            fig_exports.update_xaxes(title='HS2 (abreviado)', tickangle=-30)
            fig_exports.update_layout(
                coloraxis_showscale=False,
                height=620,
                uniformtext_minsize=9,
                uniformtext_mode='hide',
                xaxis=dict(domain=[0.0, 0.9])
            )

            fig_state_trend = None
            if not df_origin_exports.empty and {'Year', 'Month', 'State', 'Trade Value'}.issubset(df_origin_exports.columns):
                df_states_time = df_origin_exports.copy()
                df_states_time['Year'] = pd.to_numeric(df_states_time['Year'], errors='coerce')
                df_states_time = df_states_time[
                    (df_states_time['Year'] >= selected_export_year_range[0]) &
                    (df_states_time['Year'] <= selected_export_year_range[1])
                ].copy()
                df_states_time = df_states_time.dropna(subset=['Year', 'Month', 'State', 'Trade Value'])
                df_states_time['State'] = df_states_time['State'].astype(str).str.strip()
                df_states_time['state_key'] = df_states_time['State'].map(normalize_state_name)
                df_states_time = df_states_time[df_states_time['state_key'].astype(str).str.len() > 0].copy()
                if selected_region_cluster != "Todo":
                    region_keys = region_state_keys_map.get(selected_region_cluster, set())
                    df_states_time = df_states_time[df_states_time['state_key'].isin(region_keys)].copy()
                if selected_state_keys_effective:
                    df_states_time = df_states_time[df_states_time['state_key'].isin(selected_state_keys_effective)].copy()

                month_last2 = (
                    df_states_time['Month']
                    .astype(str)
                    .str.strip()
                    .str.extract(r'(\d{2})$')[0]
                )
                year_num = pd.to_numeric(df_states_time['Year'], errors='coerce')
                month_num = pd.to_numeric(month_last2, errors='coerce')

                period_from_parts = pd.to_datetime(
                    {'year': year_num, 'month': month_num, 'day': 1},
                    errors='coerce'
                )
                period_fallback = pd.to_datetime(df_states_time['Month'], errors='coerce')
                df_states_time['period'] = period_from_parts.combine_first(period_fallback)
                df_states_time = df_states_time.dropna(subset=['period'])
                df_states_time['Trade Value'] = pd.to_numeric(df_states_time['Trade Value'], errors='coerce')
                df_states_time = df_states_time.dropna(subset=['Trade Value'])

                if 'selected_month_range' in locals():
                    start_month, end_month = selected_month_range
                    start_period = pd.to_datetime(f"{start_month}-01", errors='coerce')
                    end_period = pd.to_datetime(f"{end_month}-01", errors='coerce')
                    if pd.notna(start_period) and pd.notna(end_period):
                        df_states_time = df_states_time[
                            (df_states_time['period'] >= start_period) &
                            (df_states_time['period'] <= end_period)
                        ]

                state_labels = (
                    df_states_time
                    .groupby('state_key', as_index=False)['State']
                    .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
                    .rename(columns={'State': 'State_display'})
                )
                df_state_line = (
                    df_states_time
                    .groupby(['period', 'state_key'], as_index=False)['Trade Value']
                    .sum()
                    .merge(state_labels, on='state_key', how='left')
                    .sort_values(['period', 'state_key'])
                )

                if not df_state_line.empty:
                    top_states = (
                        df_state_line
                        .groupby('state_key', as_index=False)['Trade Value']
                        .sum()
                        .sort_values('Trade Value', ascending=False)
                        .head(5)['state_key']
                        .tolist()
                    )
                    df_state_line = df_state_line[df_state_line['state_key'].isin(top_states)].copy()

                    states_all = sorted(df_state_line['State_display'].dropna().astype(str).unique().tolist())
                    state_short_map = {}
                    state_used_short = {}
                    for full_name in states_all:
                        base_short = full_name if len(full_name) <= 18 else full_name[:18].rstrip() + "..."
                        count_short = state_used_short.get(base_short, 0) + 1
                        state_used_short[base_short] = count_short
                        short_name = base_short if count_short == 1 else f"{base_short} ({count_short})"
                        state_short_map[full_name] = short_name

                    df_state_line['State corto'] = df_state_line['State_display'].map(state_short_map).fillna(df_state_line['State_display'])

                    fig_state_trend = px.line(
                        df_state_line,
                        x='period',
                        y='Trade Value',
                        color='State corto',
                        markers=True,
                        title='Evolución mensual por estado'
                    )
                    fig_state_trend.update_traces(
                        hovertemplate="Periodo: %{x|%Y-%m}<br>Trade Value: $%{y:,.0f}<extra></extra>"
                    )
                    fig_state_trend.update_yaxes(tickprefix="$", separatethousands=True)
                    fig_state_trend.update_xaxes(title='Periodo (Year + MM de Month)', type='date')
                    fig_state_trend.update_layout(height=620, legend_title_text='Estado (abreviado)')

            col_exports_line, col_exports_bar = st.columns([1.65, 1.0], gap="small")
            with col_exports_line:
                if fig_state_trend is not None:
                    st.plotly_chart(fig_state_trend, use_container_width=True)
                else:
                    st.info("No hay datos suficientes para la evolución mensual por estado.")
                    if df_origin_exports.empty:
                        st.caption("`df_origin_exports` está vacío. Verifica que la carpeta `Origen exports` y sus archivos estén incluidos en el repositorio.")
            with col_exports_bar:
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

            extra_org_rows = []
            if not df_hubs.empty and 'nombre' in df_hubs.columns:
                hubs_urls = (
                    df_hubs['url_source']
                    if 'url_source' in df_hubs.columns
                    else pd.Series([""] * len(df_hubs))
                )
                hubs_extra = pd.DataFrame({
                    'empresa': df_hubs['nombre'].astype(str),
                    'share_aprox_pct': '-',
                    'url_source': hubs_urls.astype(str)
                })
                extra_org_rows.append(hubs_extra)
            if not df_el_mexico.empty and 'empresa' in df_el_mexico.columns:
                el_urls = (
                    df_el_mexico['url_source']
                    if 'url_source' in df_el_mexico.columns
                    else pd.Series([""] * len(df_el_mexico))
                )
                el_extra = pd.DataFrame({
                    'empresa': df_el_mexico['empresa'].astype(str),
                    'share_aprox_pct': '-',
                    'url_source': el_urls.astype(str)
                })
                extra_org_rows.append(el_extra)
            if extra_org_rows:
                extra_org_table = pd.concat(extra_org_rows, ignore_index=True)
                extra_org_table = extra_org_table.dropna(subset=['empresa']).copy()
                extra_org_table['empresa'] = extra_org_table['empresa'].astype(str).str.strip()
                extra_org_table = extra_org_table[extra_org_table['empresa'].str.len() > 0]
                empresas_share_table = pd.concat([empresas_share_table, extra_org_table], ignore_index=True)

            empresas_share_table['url_source'] = empresas_share_table['url_source'].fillna('').astype(str)
            empresas_share_table['share_priority'] = np.where(
                empresas_share_table['share_aprox_pct'].astype(str).str.contains('%', na=False), 0, 1
            )
            empresas_share_table = (
                empresas_share_table
                .sort_values(['empresa', 'share_priority'], ascending=[True, True])
                .drop_duplicates(subset=['empresa'], keep='first')
                .drop(columns=['share_priority'])
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

            if not df_hubs.empty:
                hubs_map = df_hubs[['nombre', 'ciudad_hub', 'estado', 'latitud', 'longitud']].copy()
                hubs_map = hubs_map.rename(
                    columns={
                        'nombre': 'empresa',
                        'ciudad_hub': 'planta_ciudad_aprox',
                        'latitud': 'latitud_aprox',
                        'longitud': 'longitud_aprox'
                    }
                )
                hubs_map['grupo'] = 'hubs'
                hubs_map = hubs_map[['empresa', 'planta_ciudad_aprox', 'estado', 'latitud_aprox', 'longitud_aprox', 'grupo']]
                plantas_map = pd.concat([plantas_map, hubs_map], ignore_index=True)
            if not df_el_mexico.empty:
                el_mexico_map = df_el_mexico[['empresa', 'hub_mexico', 'latitud', 'longitud']].copy()
                el_mexico_map = el_mexico_map.rename(
                    columns={
                        'hub_mexico': 'planta_ciudad_aprox',
                        'latitud': 'latitud_aprox',
                        'longitud': 'longitud_aprox'
                    }
                )
                el_mexico_map['estado'] = ''
                el_mexico_map['grupo'] = 'el_mexico'
                el_mexico_map = el_mexico_map[['empresa', 'planta_ciudad_aprox', 'estado', 'latitud_aprox', 'longitud_aprox', 'grupo']]
                plantas_map = pd.concat([plantas_map, el_mexico_map], ignore_index=True)

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
                map_col_control, map_col_plot = st.columns([0.16, 0.84], gap="small")
                with map_col_control:
                    def map_toggle_spacer(height_rem: float = 5.2):
                        st.markdown(
                            f"<div style='height:{height_rem}rem;'></div>",
                            unsafe_allow_html=True
                        )

                    show_risk_heat_layer = st.toggle(
                        "Mapa de calor (riesgo)",
                        value=True,
                        key="show_risk_heat_layer_map"
                    )
                    if show_risk_heat_layer and df_estado_heat.empty:
                        st.caption("Sin datos de riesgo: revisa carpeta Municipal-Delitos-2015-2025_ene2026")
                    map_toggle_spacer()
                    show_empresas_plantas_points = st.toggle(
                        "Mostrar empresas y plantas",
                        value=True,
                        key="show_empresas_plantas_points_map"
                    )
                    map_toggle_spacer()
                    show_hubs_points = st.toggle(
                        "Mostrar hubs",
                        value=True,
                        key="show_hubs_points_map"
                    )
                    map_toggle_spacer()
                    show_el_mexico_points = st.toggle(
                        "Distribuidores estratégicos",
                        value=True,
                        key="show_el_mexico_points_map"
                    )

                bubble_map = (
                    plantas_map
                    .groupby(['latitud_aprox', 'longitud_aprox', 'region_cluster', 'grupo'], as_index=False)
                    .agg(
                        total_plantas=('empresa', 'count'),
                        empresas_unicas=('empresa', 'nunique'),
                        empresas_lista=('empresa', lambda s: "<br>".join(sorted(pd.Series(s).dropna().astype(str).unique())))
                    )
                    .sort_values('total_plantas', ascending=False)
                )
                bubble_map['ubicacion'] = bubble_map['empresas_lista']
                bubble_map['marker_size'] = np.clip(np.sqrt(bubble_map['total_plantas']) * 7.0, 8, 40)
                hubs_bubble = bubble_map[bubble_map['grupo'] == 'hubs'].copy()
                el_mexico_bubble = bubble_map[bubble_map['grupo'] == 'el_mexico'].copy()
                non_hubs_bubble = bubble_map[~bubble_map['grupo'].isin(['hubs', 'el_mexico'])].copy()
                if not show_empresas_plantas_points:
                    non_hubs_bubble = non_hubs_bubble.iloc[0:0].copy()
                if not show_hubs_points:
                    hubs_bubble = hubs_bubble.iloc[0:0].copy()
                if not show_el_mexico_points:
                    el_mexico_bubble = el_mexico_bubble.iloc[0:0].copy()

                farma_geo = non_hubs_bubble[
                    non_hubs_bubble['grupo'].isin(['plantas_20', 'plantas_80'])
                ][['latitud_aprox', 'longitud_aprox']].rename(
                    columns={'latitud_aprox': 'lat', 'longitud_aprox': 'lon'}
                )
                maquinaria_geo = non_hubs_bubble[
                    non_hubs_bubble['grupo'] == 'em_int'
                ][['latitud_aprox', 'longitud_aprox']].rename(
                    columns={'latitud_aprox': 'lat', 'longitud_aprox': 'lon'}
                )
                hubs_geo = hubs_bubble[['latitud_aprox', 'longitud_aprox']].rename(
                    columns={'latitud_aprox': 'lat', 'longitud_aprox': 'lon'}
                )
                dist_geo = el_mexico_bubble[['latitud_aprox', 'longitud_aprox']].rename(
                    columns={'latitud_aprox': 'lat', 'longitud_aprox': 'lon'}
                )

                risk_points = pd.DataFrame(columns=['lat', 'lon', 'risk_weighted'])
                fig_mexico_map = go.Figure()
                if show_risk_heat_layer and not df_estado_heat.empty:
                    estados_centroides = (
                        plantas_map[
                            plantas_map['estado'].astype(str).str.strip().ne('')
                        ][['estado', 'latitud_aprox', 'longitud_aprox']]
                        .copy()
                    )
                    if not estados_centroides.empty:
                        estados_centroides['estado_key'] = estados_centroides['estado'].map(normalize_state_name)
                        estados_heat = (
                            estados_centroides
                            .groupby(['estado', 'estado_key'], as_index=False)[['latitud_aprox', 'longitud_aprox']]
                            .mean()
                            .merge(
                                df_estado_heat[['estado_key', 'Promedio_anual_delitos']],
                                on='estado_key',
                                how='left'
                            )
                            .dropna(subset=['Promedio_anual_delitos'])
                        )
                        if not estados_heat.empty:
                            sinaloa_key = normalize_state_name('Sinaloa')
                            sinaloa_fixed = pd.DataFrame([{
                                'estado': 'Sinaloa',
                                'estado_key': sinaloa_key,
                                'latitud_aprox': 25.1721,
                                'longitud_aprox': -107.4795
                            }]).merge(
                                df_estado_heat[['estado_key', 'Promedio_anual_delitos']],
                                on='estado_key',
                                how='left'
                            )
                            estados_heat = estados_heat[estados_heat['estado_key'] != sinaloa_key]
                            estados_heat = pd.concat([estados_heat, sinaloa_fixed], ignore_index=True)
                            estados_heat = estados_heat.dropna(subset=['Promedio_anual_delitos'])

                            high_risk_boost = {
                                normalize_state_name('Guanajuato'): 2.5,
                                normalize_state_name('Baja California'): 2.5,
                                normalize_state_name('Michoacán'): 3.4,
                                normalize_state_name('Jalisco'): 1.9,
                                normalize_state_name('Chihuahua'): 2.5,
                                normalize_state_name('Sonora'): 3.4,
                                normalize_state_name('Guerrero'): 2.5,
                                normalize_state_name('Zacatecas'): 2.5,
                                normalize_state_name('Colima'): 2.5,
                                normalize_state_name('Sinaloa'): 3.0
                            }
                            estados_heat['heat_multiplier'] = estados_heat['estado_key'].map(high_risk_boost).fillna(1.0)
                            estados_heat['Promedio_anual_delitos_heat'] = (
                                estados_heat['Promedio_anual_delitos'] * estados_heat['heat_multiplier']
                            )
                            risk_points = estados_heat[['latitud_aprox', 'longitud_aprox', 'Promedio_anual_delitos_heat']].rename(
                                columns={
                                    'latitud_aprox': 'lat',
                                    'longitud_aprox': 'lon',
                                    'Promedio_anual_delitos_heat': 'risk_weighted'
                                }
                            )
                            estados_heat_sinaloa = estados_heat[estados_heat['estado_key'] == sinaloa_key].copy()
                            estados_heat_base = estados_heat[estados_heat['estado_key'] != sinaloa_key].copy()

                            if not estados_heat_base.empty:
                                fig_mexico_map.add_trace(go.Scattergeo(
                                    lat=estados_heat_base['latitud_aprox'],
                                    lon=estados_heat_base['longitud_aprox'],
                                    mode='markers',
                                    marker=dict(
                                        size=np.clip(np.sqrt(estados_heat_base['Promedio_anual_delitos_heat']) * 0.9, 16, 62),
                                        color=estados_heat_base['Promedio_anual_delitos_heat'],
                                        colorscale='YlOrRd',
                                        opacity=0.32,
                                        line=dict(width=0),
                                        colorbar=dict(title='Riesgo', x=1.12)
                                    ),
                                    text=estados_heat_base['estado'],
                                    customdata=estados_heat_base[['Promedio_anual_delitos', 'Promedio_anual_delitos_heat', 'heat_multiplier']].to_numpy(),
                                    hovertemplate="<b>%{text}</b><br>Promedio anual delitos: %{customdata[0]:,.0f}<br>Índice heatmap: %{customdata[1]:,.0f}<br>Multiplicador: x%{customdata[2]:.1f}<extra></extra>",
                                    name='Riesgo estatal'
                                ))
                            if not estados_heat_sinaloa.empty:
                                fig_mexico_map.add_trace(go.Scattergeo(
                                    lat=estados_heat_sinaloa['latitud_aprox'],
                                    lon=estados_heat_sinaloa['longitud_aprox'],
                                    mode='markers',
                                    marker=dict(
                                        size=np.clip(np.sqrt(estados_heat_sinaloa['Promedio_anual_delitos_heat']) * 0.95, 18, 66),
                                        color='#dc2626',
                                        opacity=0.52,
                                        line=dict(width=0.6, color='#991b1b')
                                    ),
                                    text=estados_heat_sinaloa['estado'],
                                    customdata=estados_heat_sinaloa[['Promedio_anual_delitos', 'Promedio_anual_delitos_heat', 'heat_multiplier']].to_numpy(),
                                hovertemplate="<b>%{text}</b><br>Promedio anual delitos: %{customdata[0]:,.0f}<br>Índice heatmap: %{customdata[1]:,.0f}<br>Multiplicador: x%{customdata[2]:.1f}<extra></extra>",
                                name='Riesgo estatal (Sinaloa)'
                            ))

                strategic_point = compute_strategic_point_cluster(
                    farma_geo, maquinaria_geo, hubs_geo, dist_geo, risk_points
                )
                strategic_risk = montecarlo_risk_forecast(
                    strategic_point=strategic_point,
                    farma_geo=farma_geo,
                    hubs_geo=hubs_geo,
                    dist_geo=dist_geo,
                    risk_points=risk_points,
                    n_simulaciones=1000
                )
                if not non_hubs_bubble.empty:
                    fig_mexico_map.add_trace(go.Scattergeo(
                        lat=non_hubs_bubble['latitud_aprox'],
                        lon=non_hubs_bubble['longitud_aprox'],
                        mode='markers',
                        marker=dict(
                            size=non_hubs_bubble['marker_size'],
                            color=non_hubs_bubble['total_plantas'],
                            colorscale='Turbo',
                            cmin=float(non_hubs_bubble['total_plantas'].min()),
                            cmax=float(non_hubs_bubble['total_plantas'].max()),
                            opacity=0.85,
                            line=dict(width=1.1, color='#111827'),
                            colorbar=dict(title='Plantas')
                        ),
                        text=non_hubs_bubble['ubicacion'],
                        customdata=non_hubs_bubble[['total_plantas', 'empresas_unicas', 'region_cluster']].to_numpy(),
                        hovertemplate="<b>%{text}</b><br>Total plantas: %{customdata[0]}<br>Empresas únicas: %{customdata[1]}<br>Región: %{customdata[2]}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
                        name='Plantas'
                    ))
                if not hubs_bubble.empty:
                    fig_mexico_map.add_trace(go.Scattergeo(
                        lat=hubs_bubble['latitud_aprox'],
                        lon=hubs_bubble['longitud_aprox'],
                        mode='markers',
                        marker=dict(
                            size=np.clip(hubs_bubble['marker_size'] * 1.35, 12, 50),
                            color='#2563eb',
                            opacity=0.95,
                            line=dict(width=1.3, color='#0f172a')
                        ),
                        text=hubs_bubble['ubicacion'],
                        customdata=hubs_bubble[['total_plantas', 'empresas_unicas', 'region_cluster']].to_numpy(),
                        hovertemplate="<b>%{text}</b><br>Total hubs: %{customdata[0]}<br>Organizaciones únicas: %{customdata[1]}<br>Región: %{customdata[2]}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
                        name='Hubs'
                    ))
                if not el_mexico_bubble.empty:
                    fig_mexico_map.add_trace(go.Scattergeo(
                        lat=el_mexico_bubble['latitud_aprox'],
                        lon=el_mexico_bubble['longitud_aprox'],
                        mode='markers',
                        marker=dict(
                            size=np.clip(el_mexico_bubble['marker_size'] * 1.25, 10, 46),
                            color='#f59e0b',
                            opacity=0.92,
                            line=dict(width=1.2, color='#7c2d12')
                        ),
                        text=el_mexico_bubble['ubicacion'],
                        customdata=el_mexico_bubble[['total_plantas', 'empresas_unicas', 'region_cluster']].to_numpy(),
                        hovertemplate="<b>%{text}</b><br>Total puntos EL México: %{customdata[0]}<br>Organizaciones únicas: %{customdata[1]}<br>Región: %{customdata[2]}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>",
                        name='Distribuidores Estratégicos'
                    ))
                if strategic_point is not None:
                    fig_mexico_map.add_trace(go.Scattergeo(
                        lat=[strategic_point['lat']],
                        lon=[strategic_point['lon']],
                        mode='markers',
                        marker=dict(size=16, color='#16a34a', symbol='star', line=dict(width=1.0, color='#14532d')),
                        hovertemplate=f"<b>Punto estratégico</b><br>Lat: {strategic_point['lat']:.4f}<br>Lon: {strategic_point['lon']:.4f}<extra></extra>",
                        name='Punto estratégico'
                    ))

                fig_mexico_map.update_layout(
                    title='Mapa estratégico'
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
                fig_mexico_map.update_layout(
                    height=640,
                    margin=dict(l=10, r=10, t=60, b=10),
                    legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0)
                )
                with map_col_plot:
                    st.plotly_chart(fig_mexico_map, use_container_width=True)
                if strategic_risk is not None:
                    st.caption("Pronóstico Monte Carlo del punto estratégico")
                    mc1, mc2, mc3 = st.columns([1, 1, 1], gap="large")
                    with mc1:
                        st.metric("Calificación promedio", f"{strategic_risk['Calificación Promedio']:.2f}")
                    with mc2:
                        st.metric("Riesgo (desviación)", f"{strategic_risk['Riesgo (Desviación)']:.2f}")
                    with mc3:
                        st.metric("P95 pesimista", f"{strategic_risk['P95 (Escenario Pesimista)']:.2f}")

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
                    df_left_table = pd.DataFrame(columns=['Year', 'Month', 'HS2 4 Digit', 'HS4 4 Digit', 'Trade Value', 'Share'])

                st.dataframe(
                    df_left_table,
                    use_container_width=True,
                    hide_index=True
                )
            with col_tab_2:
                empresas_share_table_right = empresas_share_table[
                    empresas_share_table['url_source'].astype(str).str.strip().ne('')
                ].copy()
                st.dataframe(
                    empresas_share_table_right[['empresa', 'share_aprox_pct', 'url_source']].rename(columns={'url_source': 'url'}),
                    use_container_width=True,
                    hide_index=True
                )
    except Exception as e:
        st.error(f"Error al cargar exportaciones: {e}")

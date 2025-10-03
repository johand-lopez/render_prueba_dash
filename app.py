import dash
from dash import dcc, html, Input, Output, dash_table
import dash_leaflet as dl
import dash_leaflet.express as dlx
import pandas as pd
import geopandas as gpd
import plotly.express as px
import json
import numpy as np
import branca.colormap as cm
import dash_bootstrap_components as dbc

# =============================
#   Mortalidad en Antioquia – Dash
# =============================

# Usamos un tema de Bootstrap (puedes probar otros: FLATLY, CYBORG, LUX...)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# =============================
# 1) Lectura de datos
# =============================
ruta_dataset = "datos/Mortalidad_General_en_el_departamento_de_Antioquia_desde_2005_20250915.csv"
ruta_shapefile = "datos/MGN_MPIO_POLITICO.shp"

# Dataset principal
dataset = pd.read_csv(
    ruta_dataset,
    dtype={"CodigoMunicipio": "string"}
)
# En caso de que 'Año' venga como string, lo normalizamos a numérico
if dataset["Año"].dtype != "int64" and dataset["Año"].dtype != "float64":
    dataset["Año"] = pd.to_numeric(dataset["Año"], errors="coerce")

# Shapefile (engine='pyogrio' mejora compatibilidad en Render)
dataset_shapefile = gpd.read_file(ruta_shapefile, engine="pyogrio")
dataset_shapefile = dataset_shapefile[dataset_shapefile["DPTO_CCDGO"] == "05"]
dataset_shapefile = dataset_shapefile[["MPIO_CDPMP", "MPIO_CNMBR", "geometry"]].to_crs(epsg=4326)
dataset_shapefile["MPIO_CDPMP"] = dataset_shapefile["MPIO_CDPMP"].astype("string")

# Seleccionamos columnas relevantes y normalizamos tipos
dataset_final = dataset[["NombreMunicipio", "CodigoMunicipio", "NombreRegion", "Año", "NumeroCasos", "TasaXMilHabitantes"]].copy()
dataset_final["CodigoMunicipio"] = dataset_final["CodigoMunicipio"].astype("string")

# Merge geo + datos
df_merge = dataset_shapefile.merge(
    dataset_final, left_on="MPIO_CDPMP", right_on="CodigoMunicipio", how="inner"
)

# Lista de años para los dropdowns
lista_anios = ["Todos los años"] + sorted(df_merge["Año"].dropna().unique().tolist())

# =============================
# 1.1) Agregados precalculados para "Todos los años"
# =============================
# (Mejora de rendimiento: se evita agrupar en cada callback)
df_tasa_all = (
    df_merge
    .groupby(["NombreMunicipio", "CodigoMunicipio", "NombreRegion"], as_index=False)
    .agg(TasaXMilHabitantes=("TasaXMilHabitantes", "mean"))
    .merge(dataset_shapefile[["MPIO_CDPMP", "geometry"]], left_on="CodigoMunicipio", right_on="MPIO_CDPMP", how="left")
    .drop(columns=["MPIO_CDPMP"])
)
df_tasa_all = gpd.GeoDataFrame(df_tasa_all, geometry="geometry", crs="EPSG:4326")

df_casos_all = (
    df_merge
    .groupby(["NombreMunicipio", "CodigoMunicipio", "NombreRegion"], as_index=False)
    .agg(NumeroCasos=("NumeroCasos", "sum"))
    .merge(dataset_shapefile[["MPIO_CDPMP", "geometry"]], left_on="CodigoMunicipio", right_on="MPIO_CDPMP", how="left")
    .drop(columns=["MPIO_CDPMP"])
)
df_casos_all = gpd.GeoDataFrame(df_casos_all, geometry="geometry", crs="EPSG:4326")

# =============================
#   Layout con Bootstrap
# =============================
app.layout = dbc.Container([
    dcc.Tabs([

        # ----- Contexto -----
        dcc.Tab(label="Contexto", children=[
            dbc.Container([
                html.H2("Contexto del proyecto", className="mt-4"),
                html.P("Este proyecto realiza un análisis georreferenciado de la mortalidad en el departamento de Antioquia, "
                       "a partir de registros municipales de defunciones ocurridas entre 2005 y 2021. El trabajo combina datos "
                       "estadísticos (número de casos de defunción y tasa de mortalidad por mil habitantes) con herramientas de "
                       "análisis espacial, permitiendo visualizar patrones y diferencias entre municipios y subregiones.",
                       className="fs-5"),

                html.H3("Objetivo del análisis", className="mt-3"),
                html.P("El objetivo de este trabajo es integrar y analizar la información de mortalidad en el departamento de Antioquia "
                       "de manera espacial, utilizando herramientas de georreferenciación. A partir de los datos de defunciones y de la "
                       "tasa de mortalidad por cada mil habitantes en cada municipio, junto con las geometrías oficiales de los límites "
                       "municipales, se busca:", className="fs-5"),

                html.Ul([
                    html.Li("Visualizar la distribución espacial de la mortalidad en los municipios de Antioquia."),
                    html.Li("Identificar patrones territoriales que puedan reflejar diferencias en las condiciones de salud, acceso a servicios médicos o características demográficas."),
                    html.Li("Generar mapas coropléticos y otras representaciones gráficas que faciliten la comprensión de las áreas con mayor o menor riesgo de mortalidad.")
                ], className="fs-5"),

                html.H3("Fuente del dataset", className="mt-3"),
                html.P("Los datos utilizados en este proyecto provienen del portal oficial de "
                       "Datos Abiertos de Colombia.", className="fs-5"),

                html.A("https://www.datos.gov.co/Salud-y-Protecci-n-Social/Mortalidad-General-en-el-departamento-de-Antioquia/fuc4-tvui/about_data",
                       href="https://www.datos.gov.co/Salud-y-Protecci-n-Social/Mortalidad-General-en-el-departamento-de-Antioquia/fuc4-tvui/about_data",
                       target="_blank", className="fs-5 text-primary"),

                html.Br(),
                html.Br(),
                html.P("Autor: Johan David Diaz Lopez", className="fw-bold fs-5")
            ], fluid=True)
        ]),

        # ----- Tabla de Datos -----
        dcc.Tab(label="Tabla de Datos", children=[
            dash_table.DataTable(
                id="tabla_merge",
                data=df_merge.drop(columns="geometry").to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_merge.drop(columns="geometry").columns],
                page_size=15,
                style_table={"overflowX": "auto"},
                virtualization=True,
                fixed_rows={"headers": True}
            )
        ]),

        # ----- Estadísticas -----
        dcc.Tab(label="Estadísticas descriptivas", children=[
            dash_table.DataTable(
                id="tabla_summary",
                columns=[{"name": "Variable", "id": "Variable"},
                         {"name": "Estadístico", "id": "Estadistico"},
                         {"name": "Valor", "id": "Valor"}],
                style_table={"overflowX": "auto"},
                style_cell={"fontSize": 12}
            )
        ]),

        # ----- Tasa -----
        dcc.Tab(label="Tasa de mortalidad", children=[
            dcc.Tabs([
                dcc.Tab(label="Mapa interactivo", children=[
                    html.Label("Seleccione un año:"),
                    dcc.Dropdown(id="anio_tasa", options=[{"label": i, "value": i} for i in lista_anios],
                                 value="Todos los años"),
                    html.Div(id="mapa_tasa")
                ]),
                dcc.Tab(label="Top 10 más altos", children=[
                    html.Label("Seleccione un año:"),
                    dcc.Dropdown(id="anio_top_tasa_alta", options=[{"label": i, "value": i} for i in lista_anios],
                                 value="Todos los años"),
                    dcc.Graph(id="plot_top10_tasa_alta")
                ]),
                dcc.Tab(label="Top 10 más bajos", children=[
                    html.Label("Seleccione un año:"),
                    dcc.Dropdown(id="anio_top_tasa_baja", options=[{"label": i, "value": i} for i in lista_anios],
                                 value="Todos los años"),
                    dcc.Graph(id="plot_top10_tasa_baja")
                ])
            ])
        ]),

        # ----- Defunciones -----
        dcc.Tab(label="Número de defunciones", children=[
            dcc.Tabs([
                dcc.Tab(label="Mapa interactivo", children=[
                    html.Label("Seleccione un año:"),
                    dcc.Dropdown(id="anio_casos", options=[{"label": i, "value": i} for i in lista_anios],
                                 value="Todos los años"),
                    html.Div(id="mapa_casos")
                ]),
                dcc.Tab(label="Top 10 más altos", children=[
                    html.Label("Seleccione un año:"),
                    dcc.Dropdown(id="anio_top_casos_alto", options=[{"label": i, "value": i} for i in lista_anios],
                                 value="Todos los años"),
                    dcc.Graph(id="plot_top10_casos_alto")
                ]),
                dcc.Tab(label="Top 10 más bajos", children=[
                    html.Label("Seleccione un año:"),
                    dcc.Dropdown(id="anio_top_casos_bajo", options=[{"label": i, "value": i} for i in lista_anios],
                                 value="Todos los años"),
                    dcc.Graph(id="plot_top10_casos_bajo")
                ])
            ])
        ])
    ])
], fluid=True)

# =============================
#   Callbacks
# =============================

# ---- Resumen (lo dejé con tu lógica original; solo podría haberse hecho con describe)
@app.callback(
    Output("tabla_summary", "data"),
    Input("tabla_summary", "id")
)
def update_summary(_):
    def resumen(x):
        return {
            "Mínimo": x.min(),
            "1er Cuartil": x.quantile(0.25),
            "Mediana": x.median(),
            "Media": x.mean(),
            "3er Cuartil": x.quantile(0.75),
            "Máximo": x.max()
        }

    df = []
    for col in ["NumeroCasos", "TasaXMilHabitantes"]:
        stats = resumen(df_merge[col])
        for k, v in stats.items():
            df.append({"Variable": col, "Estadistico": k, "Valor": round(float(v), 2)})
    return df

# =============================
#   Helpers para choropleth
# =============================
def _bins_from_values(values, k=7):
    # Cortes por cuantiles (mejor lectura en distribuciones sesgadas)
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return [0, 1]  # fallback
    qs = np.linspace(0, 1, k + 1)
    bins = np.unique(np.quantile(vals, qs))
    if len(bins) < 4:  # Aseguramos al menos 4 cortes
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        if vmin == vmax:
            vmax = vmin + 1.0
        bins = np.linspace(vmin, vmax, 5)
    return bins.tolist()

def _choropleth_from_gdf(gdf, value_col, colorscale_linear):
    """
    Devuelve (layer, colorbar) para un GeoDataFrame 'gdf', mapeando 'value_col'
    con colores discretizados (step) a partir de una escala lineal de 'branca'.
    """
    # GeoJSON
    features = json.loads(gdf.to_json())["features"]
    geojson = {"type": "FeatureCollection", "features": features}

    # Bins y escala de color en pasos
    vals = gdf[value_col].astype(float).to_numpy()
    bins = _bins_from_values(vals, k=7)
    step_cmap = colorscale_linear.to_step(n=len(bins)-1)
    colorscale = step_cmap.colors  # lista de colores hex

    # Estilos base y hover
    style = dict(weight=0.8, opacity=1, color="#111111", fillOpacity=0.75)
    hoverStyle = dict(weight=2, color="#222222", fillOpacity=0.95)

    # style function generada por dlx (lado JS)
    style_fn = dlx.choropleth(
        data=geojson,
        colorscale=colorscale,
        classes=bins,
        color_prop=value_col,
        style=style,
        nan_color="#f0f0f0"
    )

    layer = dl.GeoJSON(
        data=geojson,
        options=dict(style=style_fn),
        hoverStyle=hoverStyle,
        zoomToBounds=True
    )

    colorbar = dl.Colorbar(
        colorscale=colorscale,
        width=20, height=180,
        min=min(bins), max=max(bins)
    )
    return layer, colorbar

# ---- Mapas (NUEVO: choropleth real con degradé)
@app.callback(
    Output("mapa_tasa", "children"),
    Input("anio_tasa", "value")
)
def update_mapa_tasa(anio):
    if anio == "Todos los años":
        gdf = df_tasa_all.copy()
    else:
        gdf = df_merge.loc[df_merge["Año"] == anio, ["NombreMunicipio", "CodigoMunicipio", "NombreRegion", "TasaXMilHabitantes", "geometry"]].copy()
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.dropna(subset=["TasaXMilHabitantes"])

    layer, cbar = _choropleth_from_gdf(gdf, "TasaXMilHabitantes", cm.linear.YlOrRd_09)

    # Etiqueta de la barra
    cbar.unit = "Tasa por mil"

    return dl.Map(
        children=[dl.TileLayer(), layer, cbar],
        style={"width": "100%", "height": "600px"},
        center=[6.5, -75.5], zoom=7
    )

@app.callback(
    Output("mapa_casos", "children"),
    Input("anio_casos", "value")
)
def update_mapa_casos(anio):
    if anio == "Todos los años":
        gdf = df_casos_all.copy()
    else:
        gdf = df_merge.loc[df_merge["Año"] == anio, ["NombreMunicipio", "CodigoMunicipio", "NombreRegion", "NumeroCasos", "geometry"]].copy()
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
    gdf = gdf.dropna(subset=["NumeroCasos"])

    layer, cbar = _choropleth_from_gdf(gdf, "NumeroCasos", cm.linear.Blues_09)

    # Etiqueta de la barra
    cbar.unit = "Número de casos"

    return dl.Map(
        children=[dl.TileLayer(), layer, cbar],
        style={"width": "100%", "height": "600px"},
        center=[6.5, -75.5], zoom=7
    )

# ---- Gráficos Top 10 (idénticos a los tuyos)
@app.callback(
    Output("plot_top10_tasa_alta", "figure"),
    Input("anio_top_tasa_alta", "value")
)
def plot_top10_tasa_alta(anio):
    df = df_merge if anio == "Todos los años" else df_merge[df_merge["Año"] == anio]
    df = df.groupby("NombreMunicipio")["TasaXMilHabitantes"].mean().nlargest(10).reset_index()
    return px.bar(df, x="TasaXMilHabitantes", y="NombreMunicipio", orientation="h",
                  title="Top 10 municipios con mayor tasa de mortalidad", color="TasaXMilHabitantes")

@app.callback(
    Output("plot_top10_tasa_baja", "figure"),
    Input("anio_top_tasa_baja", "value")
)
def plot_top10_tasa_baja(anio):
    df = df_merge if anio == "Todos los años" else df_merge[df_merge["Año"] == anio]
    df = df.groupby("NombreMunicipio")["TasaXMilHabitantes"].mean().nsmallest(10).reset_index()
    return px.bar(df, x="TasaXMilHabitantes", y="NombreMunicipio", orientation="h",
                  title="Top 10 municipios con menor tasa de mortalidad", color="TasaXMilHabitantes")

@app.callback(
    Output("plot_top10_casos_alto", "figure"),
    Input("anio_top_casos_alto", "value")
)
def plot_top10_casos_alto(anio):
    df = df_merge if anio == "Todos los años" else df_merge[df_merge["Año"] == anio]
    df = df.groupby("NombreMunicipio")["NumeroCasos"].sum().nlargest(10).reset_index()
    return px.bar(df, x="NumeroCasos", y="NombreMunicipio", orientation="h",
                  title="Top 10 municipios con mayor número de defunciones", color="NumeroCasos")

@app.callback(
    Output("plot_top10_casos_bajo", "figure"),
    Input("anio_top_casos_bajo", "value")
)
def plot_top10_casos_bajo(anio):
    df = df_merge if anio == "Todos los años" else df_merge[df_merge["Año"] == anio]
    df = df.groupby("NombreMunicipio")["NumeroCasos"].sum().nsmallest(10).reset_index()
    return px.bar(df, x="NumeroCasos", y="NombreMunicipio", orientation="h",
                  title="Top 10 municipios con menor número de defunciones", color="NumeroCasos")

# =============================
#   Lanzar app
# =============================
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import altair as alt
    import matplotlib as plt
    import polars as pl
    import folium
    import marimo as mo
    from sklearn.cluster import DBSCAN
    import matplotlib.pyplot as plt
    RANDOM_STATE = 42
    return DBSCAN, RANDOM_STATE, folium, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🚘 Analasis de datos de transito usando algoritmos de aglomeración 🚘
    > Autor: Jesus Ivan Rivera Ramirez  Materia: Fundamentos de IA

    # Dataset

    Dataset: [Vehiculos involocrados en hechos de transito](https://datos.cdmx.gob.mx/dataset/hechos-de-transito-reportados-por-ssc-base-ampliada-no-comparativa/resource/13e41d24-3844-4eb3-9550-f77fa5149a12#)

    Los dataset utilizados corresponden a hechos de transito y vehiculos,
    - **Hechos de transito**: contiene información sobre los hechos de transito ocurridos en la cdmx
    - **Vehiculos**: contiene información sobre los vehiculos involucrados en hechos de transito
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Limpieza y preporcesamiento
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para los hechos de transito tomaremos en cuenta las columnas de:
    - hora_evento
    - tipo_evento
    - folio
    - latitud
    - longitud
    - tipo_de_interseccion
    - clasificacion_de_la_vialidad
    - dia
    - personas_fallecidas
    - personas_lesionadas
    """)
    return


@app.cell
def _(pd):
    hechosDf = pd.read_csv('data/transito.csv')
    hechosDf = hechosDf.dropna()
    columnas_hechos = ['hora_evento','tipo_evento','folio','latitud','longitud','tipo_de_interseccion',
    'clasificacion_de_la_vialidad','dia','personas_fallecidas','personas_lesionadas','prioridad']
    hechosDf = hechosDf[columnas_hechos].reset_index(drop=True)
    return (hechosDf,)


@app.cell
def _(hechosDf):
    df = hechosDf
    df = df[df.dia.isin(['Lunes','','Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])]
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploración
    """)
    return


@app.cell
def _(RANDOM_STATE, df, folium):
    _plot_points = df.sample(min(5000,len(df)), random_state=RANDOM_STATE)[["latitud","longitud"]]
    _center = [_plot_points["latitud"].mean(),_plot_points["longitud"].mean()]
    _map = folium.Map(location=_center, tiles='OpenStreetMap', prefer_canvas=True, zoom_start=12)
    for _lat, _lon in _plot_points.itertuples(index= False, name=None):
        folium.CircleMarker(location=[_lat,_lon], radius=3, fill=True).add_to(_map)
    _map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Puntos de mayor densidad usando DBSCAN
    Consideraciones adicionales, se transforma la
    """)
    return


@app.cell
def _(mo):
    EARTH_RADIUS = 6371.0088
    EPS_KM = mo.ui.slider(start=0.05, stop=1.0, step=0.05)
    MIN_SAMPLES = mo.ui.slider(start=10, stop=50, step=5)
    return EARTH_RADIUS, EPS_KM, MIN_SAMPLES


@app.cell(hide_code=True)
def _(EPS_KM, MIN_SAMPLES, mo):
    mo.hstack([mo.md(f"Distancia: {EPS_KM.value}km"),EPS_KM, mo.md(f"Vecinos minimos: {MIN_SAMPLES.value}"),MIN_SAMPLES], justify='start')
    return


@app.cell(hide_code=True)
def _(DBSCAN, EARTH_RADIUS, RANDOM_STATE, folium, np, plt):
    from matplotlib.colors import to_hex
    def calcAndShow(df, eps, minS):
        coords = np.radians(df[["latitud","longitud"]].to_numpy())
        dbscan = DBSCAN(eps = eps/EARTH_RADIUS, min_samples = minS, metric='haversine', algorithm='ball_tree', n_jobs=-1)
        labels = dbscan.fit_predict(coords)
        values, n_cluster = np.unique_counts(labels)
        print(f"""Numero de clusters: {len(values)},
        Ruido: {np.mean(labels==-1):.2}%
        eps: {eps}km
        min_samples: {minS}
        Registros con cluster: {np.sum(labels>=0)/len(df):.2}""")
        df['cluster'] = labels
        _plot_points = df.sample(min(5000,len(df)), random_state=RANDOM_STATE)[["latitud","longitud", 'cluster']]
        _center = [_plot_points["latitud"].mean(),_plot_points["longitud"].mean()]
        _map = folium.Map(location=_center, tiles='OpenStreetMap', prefer_canvas=True, zoom_start=12)
        palette = plt.get_cmap("viridis")
        for _lat, _lon, cluster in _plot_points.itertuples(index= False, name=None):
            color = to_hex(palette(cluster/len(values)))
            folium.CircleMarker(location=[_lat,_lon], radius=3, color = color, fill=True, tooltip=cluster, opacity=0.1 if cluster==-1 else 1,).add_to(_map)
        return _map



    return (calcAndShow,)


@app.cell
def _(EPS_KM, MIN_SAMPLES, calcAndShow, df):
    calcAndShow(df, EPS_KM.value, MIN_SAMPLES.value)
    return


@app.cell
def _(DBSCAN, EARTH_RADIUS, df, np, pd):
    coords = np.radians(df[["latitud","longitud"]].to_numpy())
    features = ['noche','transporte','fin','choque', 'peaton', 'otro']
    #for eps,minS in [(0.15,15),(0.15,20),(0.1,25)]:
    for eps,minS in [(0.15,15)]:
        dbscan = DBSCAN(eps = eps/EARTH_RADIUS, min_samples = minS, metric='haversine', algorithm='ball_tree', n_jobs=-1)
        labels = dbscan.fit_predict(coords)
        tmp = df.copy()
        tmp['cluster'] = labels
        tmp['hora'] = pd.to_datetime(tmp['hora_evento'], format='%H:%M:%S').dt.hour 
        tmp['noche'] = ((tmp['hora']>=20) | (tmp['hora']<=6)).astype('Float64')
        tmp['transporte'] = (((tmp['hora']>=7) & (tmp['hora']<=10)) | ((tmp['hora']>=17) & (tmp['hora']<=20))).astype('Float64')
        tmp['fin'] = (tmp['dia'].isin(['Sábado', 'Domingo'])).astype('Float64')
        tmp['choque'] = (tmp['tipo_evento'].isin(['ATROPELLADO'])).astype('Float64')
        tmp['peaton'] = (tmp['tipo_evento'].isin(['CHOQUE'])).astype('Float64')
        tmp['otro'] = (~tmp['tipo_evento'].isin(['ATROPELLADO', 'CHOQUE'])).astype('Float64')
        values, n_cluster = np.unique_counts(labels)
        noise = (labels==-1).mean()*100
        clustered = tmp[tmp['cluster']>=0]
        sizes = pd.Series(labels).loc[lambda x: x!=-1].value_counts()
        #print(sizes)
        print(sizes.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
        #print('clusters less than 30',round((sizes<30).mean(),3))

        hotspot = tmp.groupby('cluster', as_index=False)[features].mean()
        hotspot['hechos'] = tmp.groupby('cluster').size()
        hotspot = hotspot[hotspot['hechos']>=30]
        print(hotspot[features].describe().T)
    hotspot.reset_index()
    return features, hotspot


@app.cell
def _(features, hotspot, np, pd):
    h = hotspot.copy()
    print(h.columns)
    h = h[h['cluster']>=0]
    print(type(h))
    n = h['hechos'].astype('Int64')
    results = []
    for f in features:
        p = np.average(h[f], weights=n)
        observed_sd = h[f].std(ddof=1)
        expected = np.sqrt(np.mean(p*(1-p)/n))
        results.append({'feature':f, "observed":observed_sd, 'expected':expected,'ratio':observed_sd/expected})

    pd.DataFrame(results)
    return


if __name__ == "__main__":
    app.run()

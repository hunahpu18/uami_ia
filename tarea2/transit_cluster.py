import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt
    import folium
    from sklearn.cluster import DBSCAN
    from matplotlib.colors import to_hex
    sns.set_theme(style='whitegrid')
    RANDOM_STATE = 42
    return DBSCAN, RANDOM_STATE, alt, folium, mo, np, pd, plt, sns, to_hex


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analsis de clustering sobre datos de transito

    > Autor Jesus Ivan Rivera Ramirez
    > Materia: Fundamentos de IA

    Dataset: 🚘[Vehiculos involocrados en hechos de transito](https://datos.cdmx.gob.mx/dataset/hechos-de-transito-reportados-por-ssc-base-ampliada-no-comparativa/resource/13e41d24-3844-4eb3-9550-f77fa5149a12#)🚘

    > TODO agregar descripción del dataset
    """)
    return


@app.cell
def _(pd):
    transitDf = pd.read_csv("data/transito.csv")
    pd.DataFrame({
        "columna"  : transitDf.columns,
        "No_nulos" : transitDf.notna().sum().values,
        "Muestra"  : [transitDf[c].dropna().iloc[0] for c in transitDf.columns]
    })
    return (transitDf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Datos restringidos a la alcadia Iztapalapa.

    Columnas a utilizar:
    - latitud
    - longitud
    - dia
    - hora
    - tipo de evento
    - tipo de intersección
    - personas heridas
    - personas fallecidas
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Limpieza quitar espacios de las columnas de texto y quitar caracteres especiales
    """)
    return


@app.cell
def _(transitDf):
    transitDf
    return


@app.cell
def _(plt):
    def showCounts(df, column, label):
        df[column].value_counts().plot(kind='bar')
        plt.xlabel(label)
        plt.ylabel('Frecuencia')
        plt.show()

    return (showCounts,)


@app.cell(expand_output=True)
def _(showCounts, transitDf):
    showCounts(transitDf,'alcaldia',"Alcaldia")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Nos enfocaremos solo en los accidentes locales de la alcaldia iztapalapa
    """)
    return


@app.cell
def _(transitDf):
    iztapalapaDf = transitDf[transitDf['alcaldia']=="IZTAPALAPA"][["latitud","longitud","dia","hora_evento","tipo_de_interseccion","clasificacion_de_la_vialidad","personas_fallecidas","personas_lesionadas","tipo_evento","prioridad"]]
    iztapalapaDf.isna().sum()
    return (iztapalapaDf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Limpieza de datos, se eliminan datos nulos, se conservan solo dias
    """)
    return


@app.cell
def _(iztapalapaDf):
    iztapalapaDf['dia'].value_counts()
    return


@app.cell
def _(iztapalapaDf, np, pd):
    iztCleanDf = iztapalapaDf.dropna()
    iztCleanDf = iztCleanDf[iztCleanDf['dia'].isin(['Lunes','','Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])]
    iztCleanDf['hora_evento'] = pd.to_datetime(iztCleanDf['hora_evento'], format='%H:%M:%S').dt.hour 
    iztCleanDf['hora_sin'] = np.sin(iztCleanDf['hora_evento']*np.pi/24*2)
    iztCleanDf
    return (iztCleanDf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numero de accidentes por evento
    """)
    return


@app.cell
def _(iztCleanDf, showCounts):
    showCounts(iztCleanDf,'tipo_evento',"Tipo de evento")
    return


@app.cell
def _(iztCleanDf, showCounts):
    showCounts(iztCleanDf,'dia',"Dia de la semana")
    return


@app.cell
def _(iztCleanDf, showCounts):
    showCounts(iztCleanDf,'personas_lesionadas',"Personas lesionadas")
    return


@app.cell
def _(iztCleanDf, showCounts):
    showCounts(iztCleanDf,'personas_fallecidas',"Personas fallecidas")
    return


@app.cell
def _(iztCleanDf, showCounts):
    showCounts(iztCleanDf,'tipo_de_interseccion',"Tipo de intersección")
    return


@app.cell
def _(iztCleanDf, showCounts):
    showCounts(iztCleanDf,'hora_evento',"Hora")
    return


@app.cell
def _(iztCleanDf, showCounts):
    showCounts(iztCleanDf,'clasificacion_de_la_vialidad',"Vialidad")
    return


@app.cell
def _(iztCleanDf, showCounts):
    showCounts(iztCleanDf,'tipo_de_interseccion',"Interseccion")
    return


@app.cell(expand_output=True)
def _(RANDOM_STATE, folium, iztCleanDf):
    _plot_points = iztCleanDf.sample(10000, random_state=RANDOM_STATE)[["latitud","longitud"]]
    _center = [_plot_points["latitud"].mean(),_plot_points["longitud"].mean()]
    _map = folium.Map(location=_center, tiles='OpenStreetMap', prefer_canvas=True, zoom_start=12, zoom_control=False, scrollWheelZoom=False, dragging=False)
    for lat, lon in _plot_points.itertuples(index= False, name=None):
        folium.CircleMarker(location=[lat,lon], radius=1.5, fill=True).add_to(_map)
    _map
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DBSCAN
    """)
    return


@app.cell
def _(mo):
    EARTH_RADIUS = 6371.0088
    EPS_KM = mo.ui.slider(start=0.05, stop=1.0, step=0.05)
    MIN_SAMPLES = mo.ui.slider(start=10, stop=50, step=5)
    return EARTH_RADIUS, EPS_KM, MIN_SAMPLES


@app.cell
def _(EPS_KM, MIN_SAMPLES, mo):
    mo.hstack([mo.md(f"Distancia: {EPS_KM.value}km"),EPS_KM, mo.md(f"Vecinos minimos: {MIN_SAMPLES.value}"),MIN_SAMPLES], justify='start')
    return


@app.cell(hide_code=True)
def _(DBSCAN, EARTH_RADIUS, np):
    def calcCluster(df, minSamples: int, eps: int):
        coords = np.radians(df[["latitud","longitud"]].to_numpy())
        dbscan = DBSCAN(eps = eps/EARTH_RADIUS, min_samples = minSamples, metric='haversine', algorithm='ball_tree', n_jobs=-1)
        labels = dbscan.fit_predict(coords)
        values, n_cluster = np.unique_counts(labels)
        print(f"""Numero de clusters: {len(values)},
        Ruido: {np.mean(labels==-1):.2}%
        eps: {eps}km
        min_samples: {minSamples}
        Registros con cluster: {np.sum(labels>=0)/len(df):.2}""")
        return labels


    return (calcCluster,)


@app.cell(hide_code=True)
def _(Dataframe, RANDOM_STATE, plt, sns):
    def showCluster(df: Dataframe, labels,cluster=None):
        df['cluster'] = labels
        if(cluster):
            _subset = df.loc[df.cluster == cluster]
        else:
            _subset=df
        _subset = _subset[_subset.cluster>=0]
        _plot_points = _subset.sample(min(3000,len(_subset)), random_state=RANDOM_STATE)
        _fig, _ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(_plot_points, y='latitud', x='longitud', hue='cluster', legend=True, palette='tab20')
        _ax.set(xlabel="Latitud", ylabel="Longitud", title="Hechos de Transito en Iztapalapa")
        _ax.set_aspect('equal',adjustable='datalim')

        plt.legend(title="Cluster", loc='upper left', bbox_to_anchor=(1, 1))
        df.drop(columns=['cluster'],inplace=True)
        plt.show()


    return (showCluster,)


@app.cell(hide_code=True)
def _(Dataframe, RANDOM_STATE, folium, plt, to_hex):
    def showMap(df: Dataframe, labels,cluster=None):
        df['cluster'] = labels
        if(cluster):
            _subset = df.loc[df.cluster == cluster]
        else:
            _subset=df
        _subset = _subset[_subset['cluster']>=0]
        _plot_points = _subset.sample(min(3000,len(_subset)), random_state=RANDOM_STATE)[["latitud","longitud",'cluster']]
        _ncluster = _plot_points.loc[_plot_points.cluster>=0, 'cluster'].nunique()
        _center = [_plot_points["latitud"].mean(),_plot_points["longitud"].mean()]
        _map = folium.Map(location=_center, tiles='OpenStreetMap', prefer_canvas=True, zoom_start=12)
        palette = plt.get_cmap("viridis")
        for _lat, _lon, cluster in _plot_points.itertuples(index= False, name=None):
            color = to_hex(palette(cluster/_ncluster))
            folium.CircleMarker(location=[_lat,_lon], radius=3, color = color, fill=True, tooltip=cluster).add_to(_map)
        return _map

    return (showMap,)


@app.cell(hide_code=True)
def _(EPS_KM, MIN_SAMPLES, alt, calcCluster, iztCleanDf, np, pd):
    _labels = calcCluster(iztCleanDf,MIN_SAMPLES.value,EPS_KM.value)
    _unique, _counts = np.unique(_labels[_labels>=0], return_counts=True)
    _df = pd.DataFrame({'value':_unique,'count':_counts})
    # replace _df with your data source
    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X(field='value', type='quantitative', title='Id Cluster').scale(domainMin=0),
            y=alt.Y(field='count', type='quantitative', title='Frecuencia'),
            tooltip=[
                alt.Tooltip(field='value', format=',.0f'),
                alt.Tooltip(field='count', format=',.0f')
            ]
        )
        .properties(
            title='Puntos por cluster',
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    _chart
    return


@app.cell
def _(transitDf):
    len(transitDf.dropna())*0.1
    return


@app.cell
def _(EPS_KM, MIN_SAMPLES, calcCluster, showCluster, showMap, transitDf):
    _explo_labels = calcCluster(transitDf.dropna(),MIN_SAMPLES.value,EPS_KM.value)
    showCluster(transitDf.dropna(), _explo_labels)
    showMap(transitDf.dropna(), _explo_labels)
    return


@app.cell
def _(mo, transitDf):
    mo.ui.table(transitDf)
    return


@app.cell
def _(EPS_KM, MIN_SAMPLES, calcCluster, iztCleanDf, showCluster, showMap):
    explo_labels = calcCluster(iztCleanDf,MIN_SAMPLES.value,EPS_KM.value)
    showCluster(iztCleanDf, explo_labels)
    showMap(iztCleanDf, explo_labels)
    return (explo_labels,)


@app.cell
def _(explo_labels, mo, np):
    number = mo.ui.number(start=0, stop=len(np.unique_counts(explo_labels)[0])-2, step=1)
    return (number,)


@app.cell
def _(mo, number):
    mo.hstack([number, mo.md(f"Has value: {number.value}")])
    return


@app.cell
def _(explo_labels, iztCleanDf, number, showMap):
    showMap(iztCleanDf, explo_labels,cluster=number.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    eps: 0.2km
    min_samples: 25
    cluster 8
    """)
    return


@app.cell
def _(calcCluster, iztCleanDf, showMap):
    labels = calcCluster(iztCleanDf, 25, 0.05)
    showMap(iztCleanDf, labels)
    return (labels,)


@app.cell
def _(iztCleanDf, labels):
    iztCleanDf['cluster']=labels
    _noNoise = iztCleanDf[iztCleanDf['cluster']>=0]
    iztCleanDf.groupby(['tipo_de_interseccion'])['cluster'].nunique()
    return


@app.cell
def _(iztCleanDf, labels):
    iztCleanDf['cluster']=labels
    _noNoise = iztCleanDf[iztCleanDf['cluster']>=0]
    iztCleanDf.groupby(['hora_evento','tipo_de_interseccion'])['cluster'].nunique()
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell
def _(alt, iztCleanDf, labels):
    iztCleanDf['cluster']=labels
    _noNoise = iztCleanDf[iztCleanDf['cluster']>=0]
    alt.Chart(_noNoise).mark_bar().encode(
        x='cluster',
        y='count(dia)',
        color='dia'
    )
    return


@app.cell
def _(alt, iztCleanDf):
    _noNoise = iztCleanDf[iztCleanDf['cluster']>=0]
    alt.Chart(_noNoise).mark_bar().encode(
        x='cluster',
        y='count(hora_evento)',
        color='hora_evento'
    )
    return


@app.cell
def _(alt, iztCleanDf):
    _noNoise = iztCleanDf[iztCleanDf['cluster']>=0]
    alt.Chart(_noNoise).mark_bar().encode(
        x='cluster',
        y='count(tipo_de_interseccion)',
        color='tipo_de_interseccion'
    )
    return


@app.cell
def _(alt, iztCleanDf):
    _noNoise = iztCleanDf[iztCleanDf['cluster']>=0]
    alt.Chart(_noNoise).mark_bar().encode(
        x='cluster',
        y='count(clasificacion_de_la_vialidad)',
        color='clasificacion_de_la_vialidad'
    )
    return


@app.cell
def _(alt, iztCleanDf):
    _noNoise = iztCleanDf[iztCleanDf['cluster']>=0]
    alt.Chart(_noNoise).mark_bar().encode(
        x='cluster',
        y='count(clasificacion_de_la_vialidad)',
        color='clasificacion_de_la_vialidad'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    enfocarnos en el cluster 8
    """)
    return


@app.cell
def _(iztCleanDf, labels, showMap):
    showMap(iztCleanDf, labels, cluster = 8)
    return


@app.cell
def _(alt, secondDF):

    _columns = [
        'dia',
        'hora_evento',
        'tipo_evento',
        'clasificacion_de_la_vialidad',
        'tipo_de_interseccion',
        'personas_fallecidas',
        'personas_lesionadas',
    ]
    _charts = []

    for _c in _columns:
      _chart = (
          alt.Chart(secondDF)
          .mark_bar()
          .encode(
              x=alt.X(alt.repeat('column')),
              y=alt.Y('count():Q', title=f'Conteo ({_c})'),
              color=alt.Color(_c, title=_c),
          )
          .properties(width=150, height=150)
          .repeat(
              column=_columns,
          )
          .interactive()
      )
      _charts.append(_chart)

    final_chart = alt.vconcat(*_charts).resolve_scale(
        x='independent',  # Escalas independientes para cada variable en el eje X
        y='independent',  # Escalas independientes para el eje Y
    )

    final_chart
    return


@app.cell
def _(iztCleanDf, np):
    secondDF = iztCleanDf
    day_dict = {"Lunes":0,'Martes':1,'Miércoles':2,'Jueves':3,'Viernes':4,'Sábado':5,'Domingo':6}  
    secondDF['dia_n'] = secondDF['dia'].map(day_dict)
    secondDF['dia_n'] = np.sin(secondDF['dia_n']*np.pi/7*2)
    return (secondDF,)


@app.cell
def _(iztCleanDf, pd):
    _columns = [
        'dia_n',
        'hora_sin',
    ]
    features = pd.get_dummies(iztCleanDf[_columns], dtype=int)
    features
    return (features,)


@app.cell
def _(features, labels, secondDF, showCluster):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    for i in range(2,10):
        clustering = KMeans(n_clusters=i, random_state=42, n_init="auto")
        clusters = clustering.fit_predict(features)
        print(i,silhouette_score(features, clusters))
    clustering = KMeans(n_clusters=4, random_state=42, n_init="auto")
    clusters = clustering.fit_predict(features)

    showCluster(secondDF,clusters)
    secondDF['cluster']=labels
    secondDF['cluster_1'] = clusters
    return


@app.cell
def _(secondDF):
    secondDF[['cluster','hora_evento']].value_counts()
    return


@app.cell
def _(secondDF):
    secondDF
    return


@app.cell
def _(secondDF):
    _susbet = secondDF[secondDF['cluster']>=0]
    _susbet
    return


@app.cell
def _(alt, secondDF):
    # replace _df with your data source
    _susbet = secondDF[secondDF['cluster']>=0]
    _susbet
    _chart = (
        alt.Chart(_susbet)
        .mark_bar()
        .encode(
            x=alt.X(field='cluster', type='quantitative'),
            y=alt.Y(field='cluster_1', aggregate='count'),
            color=alt.Color(field='cluster_1', scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip(field='cluster', format=',.0f'),
                alt.Tooltip(field='cluster_1', aggregate='count', format=',.0f'),
                alt.Tooltip(field='cluster_1', format=',.0f')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    _chart
    return


@app.cell
def _(secondDF):
    _susbet = secondDF[(secondDF['cluster']>=0) &(secondDF['cluster_1']==0)]
    _susbet
    return


@app.cell
def _(alt, secondDF):
    _columns = ['dia', 'hora_evento','tipo_evento', 'clasificacion_de_la_vialidad', 'tipo_de_interseccion','personas_fallecidas','personas_lesionadas',]
    _susbet = secondDF[secondDF['cluster']>=0]
    _chart = alt.Chart(secondDF).mark_bar().encode(
        x=alt.X(alt.repeat("column")),
        y='count(cluster_1)',
        color=alt.Color("cluster_1:N", scale=alt.Scale(scheme="category10"))
    ).properties(
        width=150,
        height=150
    ).repeat(
        column=_columns
    ).interactive()

    _chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TODO
    > donde se concentra los accidentes?
    > cuando se concentran los accidentes?
    > cuando y donde?
    > que caracteristicas compartes las regiones

    1. describir el dataset
    """)
    return


if __name__ == "__main__":
    app.run()

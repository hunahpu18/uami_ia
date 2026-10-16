import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import altair as alt
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import MinMaxScaler

    return DBSCAN, alt, np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv("data/transito.csv")
    df.head()
    return (df,)


@app.cell
def _(df):
    izt_df = df
    izt_df
    return (izt_df,)


@app.cell
def _(izt_df):
    columns = ["fecha_evento","hora_evento","dia","tipo_evento","latitud","longitud","tipo_de_interseccion","clasificacion_de_la_vialidad", "personas_fallecidas","personas_lesionadas"]
    data = izt_df[columns]
    data.describe()
    return (data,)


@app.cell
def _(data):
    data.isna().sum()
    return


@app.cell
def _(RANDOM, data, np, pd):
    data_clean = data.dropna().sample(n=5000,random_state=RANDOM)
    data_clean['hora'] = np.sin(pd.to_datetime(data_clean['hora_evento'], format='%H:%M:%S').dt.hour*np.pi/24*2)
    data_clean.isna().sum()
    return (data_clean,)


@app.cell
def _(data_clean):
    data_clean.describe()
    return


@app.cell
def _(data_clean):
    data_clean
    return


@app.cell
def _(data_clean):
    _c = data_clean[["hora","tipo_evento"]].value_counts()
    return


@app.cell
def _(data_clean):
    data_clean["dia"].value_counts()
    return


@app.cell
def _(data_clean):
    data_clean["personas_fallecidas"].value_counts()
    return


@app.cell
def _(data_clean):
    data_clean[["personas_lesionadas"]].value_counts()
    return


@app.cell
def _(DBSCAN, data_clean, np):
    from sklearn.metrics.pairwise import haversine_distances
    X=data_clean[["latitud","longitud"]]

    # 2. Convert Lat/Long from Degrees to Radians
    coords_rad = np.radians(X)

    # 3. Define Parameters
    EARTH_RADIUS_KM = 6371.0088
    distance_limit_km = 1.4  # Max distance to consider points in the same neighborhood

    # Convert distance limit into radians for epsilon
    epsilon = distance_limit_km / EARTH_RADIUS_KM

    # 4. Run DBSCAN with Haversine Metric


    clustering =db = DBSCAN(
        eps=epsilon,
        min_samples=50,
        metric='haversine',
        algorithm='ball_tree'
    ).fit(coords_rad)
    clustering.labels_
    return X, clustering


@app.cell
def _(X, clustering, plt):
    plt.scatter(X["latitud"],X["longitud"],c=clustering.labels_)
    return


@app.cell
def _(clustering):
    n_clusters_ = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_clusters_
    return


@app.cell
def _(clustering, data_clean):
    data_clean["label"] = clustering.labels_
    return


@app.cell
def _(data_clean):
    data_clean["label"].value_counts()
    return


@app.cell(hide_code=True)
def _(alt, data_clean):
    # replace _df with your data source
    _chart = (
        alt.Chart(data_clean[data_clean["label"]>0])
        .mark_bar()
        .encode(
            x=alt.X(field='label', type='quantitative'),
            y=alt.Y(field='dia', type='nominal', stack=True, aggregate='count'),
            color=alt.Color(field='dia', type='nominal'),
            tooltip=[
                alt.Tooltip(field='label', format=',.0f'),
                alt.Tooltip(field='dia', aggregate='count'),
                alt.Tooltip(field='dia')
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
def _(data_clean):
    data_clean
    return


@app.cell
def _():
    import folium

    # Initialize the map centered around a location
    m = folium.Map(location=[38.0, -95.0], zoom_start=4)

    # List of coordinates: [(Lat, Lon, Name)]
    locations = [
        (40.7128, -74.0060, "New York"),
        (34.0522, -118.2437, "Los Angeles"),
        (41.8781, -87.6298, "Chicago"),
    ]

    # Add markers for each latitude/longitude point
    for lat, lon, name in locations:
        folium.Marker(
            location=[lat, lon],
            popup=name,
            tooltip=f"Click for {name}"
        ).add_to(m)

    # Save to an interactive HTML file
    m.show_in_browser()
    return


if __name__ == "__main__":
    app.run()

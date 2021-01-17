import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import umap

from MulticoreTSNE import MulticoreTSNE as TSNE

# @st.cache is speedup option

@st.cache
def standardize(df):
    X = df.values
    X_std = StandardScaler().fit_transform(X)
    return pd.DataFrame(X_std, columns=df.columns)

@st.cache
def do_pca(df):
    pca = PCA(n_components=3) 
    embedding = pca.fit_transform(df.values)
    return pd.DataFrame(embedding)

@st.cache
def do_umap(df):
    umap_ = umap.UMAP(n_components=3)
    embedding = umap_.fit_transform(df.values)
    return pd.DataFrame(embedding)

@st.cache
def do_tsne(df):
    tsne = TSNE(n_components=3)
    embedding = tsne.fit_transform(df.values)
    return pd.DataFrame(embedding)

def reduce_dimension(df, method='PCA'):
    if method == 'PCA':
        df = do_pca(df)
    elif method == 'UMAP':
        df = do_umap(df)
    elif method == 't-SNE':
        df = do_tsne(df)
    return df

@st.cache
def do_kmeans(df, num_c):
    kmeans = KMeans(n_clusters=num_c, random_state=100)
    clusters = kmeans.fit(df.values)
    return clusters.labels_

def clustering(df, method='k-means', num_c=4):
    if method == 'k-means':
        labels = do_kmeans(df, num_c)
    return labels


def main():

    st.title('csv visualizer')
    uploaded_file = st.sidebar.file_uploader("ファイルアップロード", type='csv') 

    if uploaded_file is not None:

        df_raw = pd.read_csv(uploaded_file)

        # sidemenu
        st.sidebar.title("Dimensionality reduction Menu")
        dimensionality_reduction_method = st.sidebar.selectbox(
            "Dimensionality reduction method:", ["PCA", "UMAP", "t-SNE"]
        )
        st.sidebar.title("Clustering Menu")
        clustering_method = st.sidebar.selectbox(
            "Clustering method:", ["k-means"]
        )
        max_cluster = min(len(df_raw), 10)
        num_cluster = st.sidebar.slider('cluster num',  min_value=1, max_value=max_cluster, step=1, value=4)

        st.header(dimensionality_reduction_method)
    
        df_std = standardize(df_raw)
        labels = clustering(df_std, clustering_method, num_cluster)
        df_reduce_dim = reduce_dimension(df_std, dimensionality_reduction_method)

        series_label = pd.Series(labels)
        fig = go.Figure()
        for i in series_label.unique():
            tmp = df_reduce_dim.loc[series_label == i].iloc[:, 0:3]
            fig.add_trace(go.Scatter3d(x=tmp[0], y=tmp[1], z=tmp[2],
                            mode='markers',
                            marker=dict(
                                size=2,
                            )))
        fig.update_layout(
        width=750, height=750,
        # margin=dict(l=40, r=40, b=40, t=40)
        )
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
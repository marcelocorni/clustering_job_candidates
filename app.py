import pandas as pd
from ydata_profiling import ProfileReport
import streamlit as st
import hashlib
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff
import IPython
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(layout="wide")

# Função para gerar o URL do Gravatar a partir do e-mail
def get_gravatar_url(email, size=100):
    email_hash = hashlib.md5(email.strip().lower().encode('utf-8')).hexdigest()
    gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?s={size}"
    return gravatar_url

# Definir o e-mail e o tamanho da imagem
email = "marcelo@desenvolvedor.net"  # Substitua pelo seu e-mail
size = 200  # Tamanho da imagem

# Layout principal com colunas
col1, col2 = st.columns([1, 3])

# Obter o URL do Gravatar
gravatar_url = get_gravatar_url(email, size)

# Layout principal com colunas
col1, col2 = st.columns([1, 3])

# Conteúdo da coluna esquerda
with col1:
    st.markdown(
        f"""
        <div style="text-align: right;">
            <img src="{gravatar_url}" alt="Gravatar" style="width: 250px;">
        </div>
        """,
        unsafe_allow_html=True
    )
# Conteúdo da coluna direita
with col2:
    st.title("Análise de Dados e Agrupamento de Candidatos a Emprego")
    st.write("## Marcelo Corni Alves")
    st.write("Agosto/2024")
    st.write("Disciplina: Mineração de Dados")


# Adicionar espaço entre as colunas e o expander
st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

@st.cache_data
def carregar_dados():
    return pd.read_csv("data/trabalho4_MARCELO-CORNI.csv")

# Carregar os dados
data = carregar_dados()

st.header("Dados Carregados")
st.write(data.head())

# Criar o relatório
pr = ProfileReport(data)

# Exibir o relatório no Streamlit
with st.expander("Relatório dos dados (YData Profiling)", expanded=False):
    st_profile_report(pr)

#with st.expander("Agrupamento dos dados", expanded=False):

# Tratar valores ausentes rotulados como "MD"
data.replace("MD", pd.NA, inplace=True)
data.dropna(inplace=True)

# Remover duplicados
data.drop_duplicates(inplace=True)

# Mapear a variável dependente "Performance" para valores numéricos
performance_mapping = {'LP': 0, 'MP': 1, 'BP': 2}
data['Performance'] = data['Performance'].map(performance_mapping)

# Remover a variável dependente "Performance" dos dados para clustering
data_without_performance = data.drop(columns=['Performance'])

# Convertendo colunas categóricas em numéricas usando One-Hot Encoding
data_encoded = pd.get_dummies(data_without_performance)

# Escalar os dados
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Calcular a variância dos dados brutos
raw_variance = data_encoded.var().mean()

# Aplicar PCA para redução de dimensionalidade
pca = PCA(n_components=3)  # Mudamos para 3 componentes para visualização 3D
data_pca = pca.fit_transform(data_scaled)
explained_variance = pca.explained_variance_ratio_

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['KMeans_Labels'] = kmeans.fit_predict(data_pca)

# Aplicar Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3)
data['Agg_Labels'] = agg_clustering.fit_predict(data_pca)

# Aplicar Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
data['GMM_Labels'] = gmm.fit_predict(data_pca)

# Avaliação intrínseca
kmeans_silhouette = silhouette_score(data_pca, data['KMeans_Labels'])
agg_silhouette = silhouette_score(data_pca, data['Agg_Labels'])
gmm_silhouette = silhouette_score(data_pca, data['GMM_Labels'])

# Avaliação extrínseca (usando "Performance")
def generate_confusion_matrix(true_labels, cluster_labels, title):
    cm = confusion_matrix(true_labels, cluster_labels)
    unique_clusters = sorted(set(cluster_labels))
    
    x_labels = [f'Grupo {i}' for i in unique_clusters]
    y_labels = ['LP', 'MP', 'BP']

    z_text = [[str(y) for y in x] for x in cm]

    fig = ff.create_annotated_heatmap(cm, x=x_labels, y=y_labels, annotation_text=z_text, colorscale='Blues')
    fig.update_layout(title_text=title, xaxis_title="Predicted", yaxis_title="Actual")
    return fig

kmeans_confusion = generate_confusion_matrix(data['Performance'], data['KMeans_Labels'], "K-Means Confusion Matrix")
agg_confusion = generate_confusion_matrix(data['Performance'], data['Agg_Labels'], "Agglomerative Clustering Confusion Matrix")
gmm_confusion = generate_confusion_matrix(data['Performance'], data['GMM_Labels'], "Gaussian Mixture Model Confusion Matrix")

# Visualização dos clusters PCA
def plot_pca_clusters(data, labels, title):
    df = pd.DataFrame(data, columns=['PC1', 'PC2'])
    df['Cluster'] = labels
    fig = px.scatter(df, x='PC1', y='PC2', color='Cluster', title=title)
    return fig

def plot_pca_clusters_3d(data, labels, title):
    df = pd.DataFrame(data, columns=['PC1', 'PC2', 'PC3'])
    df['Cluster'] = labels
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='Cluster', title=title)
    return fig

# Mostrar análises descritivas usando Plotly
def plot_group_descriptions(groups, title):
    groups = groups.rename(columns={'level_0': 'Metric', 'level_1': 'Statistic'})
    groups_melted = groups.melt(id_vars=['Metric', 'Statistic'], var_name='Cluster', value_name='Value')
    fig = px.bar(groups_melted, x='Metric', y='Value', color='Cluster', title=title, barmode='group', facet_col='Statistic', facet_col_wrap=2)
    return fig

    
with st.expander("PCA (Análise de Componentes Principais)", expanded=False):
    st.write("Proporção da Variância Explicada por Cada Componente Principal:", explained_variance)
    pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2', 'PC3'])
    st.write("Dados Transformados pelo PCA:")
    st.write(pca_df.head())

    # Gráfico de Pareto da Variância Explicada
    st.header("Gráfico de Pareto da Variância Explicada")
    fig_pareto = px.bar(
        x=['PC1', 'PC2', 'PC3'],
        y=explained_variance,
        labels={'x': 'Componentes Principais', 'y': 'Proporção da Variância Explicada'},
        title='Variância Explicada pelos Componentes Principais'
    )
    st.plotly_chart(fig_pareto)

    # Comparação da variância
    st.header("Comparação da Variância")
    st.write("Variância Média dos Dados Brutos:", raw_variance)
    st.write("Proporção da Variância Explicada pelos Componentes Principais do PCA:", explained_variance)

with st.expander("K-Means", expanded=False):
    st.header("Resultados de Agrupamento")
    st.write("K-Means Silhouette Score:", kmeans_silhouette)
    # Gráfico de Cotovelo para K-Means
    st.header("Gráfico de Cotovelo para K-Means")
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_pca)
        inertia.append(kmeans.inertia_)

    fig_elbow = px.line(x=K, y=inertia, labels={'x': 'Número de Clusters', 'y': 'Inércia'},
                        title='Gráfico de Cotovelo para K-Means')
    st.plotly_chart(fig_elbow)

    st.subheader("Matriz de Confusão - K-Means")
    st.plotly_chart(kmeans_confusion)

    # Análise descritiva dos grupos
    kmeans_groups = data.groupby('KMeans_Labels').describe().T.reset_index()
    st.header("Análise Descritiva dos Grupos")
    st.subheader("K-Means")
    fig_kmeans_groups = plot_group_descriptions(kmeans_groups, "K-Means Group Descriptions")
    st.plotly_chart(fig_kmeans_groups)

    st.header("Visualização PCA dos Clusters")
    st.subheader("K-Means Clusters 2D")
    fig_kmeans_pca = plot_pca_clusters(data_pca[:, :2], data['KMeans_Labels'], "K-Means PCA Clusters")
    st.plotly_chart(fig_kmeans_pca)

    st.subheader("K-Means Clusters 3D")
    fig_kmeans_pca_3d = plot_pca_clusters_3d(data_pca, data['KMeans_Labels'], "K-Means PCA Clusters 3D")
    st.plotly_chart(fig_kmeans_pca_3d)

with st.expander("Agglomerative Clustering", expanded=False):
    st.header("Resultados de Agrupamento")
    st.write("Agglomerative Clustering Silhouette Score:", agg_silhouette)
    st.subheader("Matriz de Confusão - Agglomerative Clustering")
    st.plotly_chart(agg_confusion)

    # Análise descritiva dos grupos
    agg_groups = data.groupby('Agg_Labels').describe().T.reset_index()
    st.header("Análise Descritiva dos Grupos")
    st.subheader("Agglomerative Clustering")
    fig_agg_groups = plot_group_descriptions(agg_groups, "Agglomerative Clustering Group Descriptions")
    st.plotly_chart(fig_agg_groups)

    st.header("Visualização PCA dos Clusters")
    st.subheader("Agglomerative Clustering Clusters 2D")
    fig_agg_pca = plot_pca_clusters(data_pca[:, :2], data['Agg_Labels'], "Agglomerative Clustering PCA Clusters")
    st.plotly_chart(fig_agg_pca)

    st.subheader("Agglomerative Clustering Clusters 3D")
    fig_agg_pca_3d = plot_pca_clusters_3d(data_pca, data['Agg_Labels'], "Agglomerative Clustering PCA Clusters 3D")
    st.plotly_chart(fig_agg_pca_3d)


with st.expander("Gaussian Mixture Model", expanded=False):
    st.header("Resultados de Agrupamento")
    st.write("Gaussian Mixture Model Silhouette Score:", gmm_silhouette)
    st.subheader("Matriz de Confusão - Gaussian Mixture Model")
    st.plotly_chart(gmm_confusion)

    # Análise descritiva dos grupos
    gmm_groups = data.groupby('GMM_Labels').describe().T.reset_index()
    st.header("Análise Descritiva dos Grupos")
    st.subheader("Gaussian Mixture Model")
    fig_gmm_groups = plot_group_descriptions(gmm_groups, "Gaussian Mixture Model Group Descriptions")
    st.plotly_chart(fig_gmm_groups)
    
    st.header("Visualização PCA dos Clusters")
    st.subheader("Gaussian Mixture Model Clusters 2D")
    fig_gmm_pca = plot_pca_clusters(data_pca[:, :2], data['GMM_Labels'], "Gaussian Mixture Model PCA Clusters")
    st.plotly_chart(fig_gmm_pca)

    st.subheader("Gaussian Mixture Model Clusters 3D")
    fig_gmm_pca_3d = plot_pca_clusters_3d(data_pca, data['GMM_Labels'], "Gaussian Mixture Model PCA Clusters 3D")
    st.plotly_chart(fig_gmm_pca_3d)
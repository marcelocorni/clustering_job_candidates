

class corni_agglomerative_clustering:
    def __init__(self,st, data, n_clusters=2,  metric='euclidean', linkage='ward' ):        
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        self.data = data
        self.st = st        

    def apply(self):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        try:
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters, metric=self.metric, linkage=self.linkage)
            self.model.fit_predict(self.data)
            self.labels = self.model.labels_
            self.silhouette_score = silhouette_score(self.data, self.labels)
        except Exception as e:
            self.st.write(f'Erro no Agglomerative com {self.n_clusters} clusters, métrica {self.metric} e linkage {self.linkage}')
    
    def plot_dendrogram_matplotlib(self, cut_value=0.5):
            
            from matplotlib import pyplot as plt
            from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

            plt.style.use('dark_background')

            # Gerar o linkage matrix
            Z = linkage(self.data, method=self.linkage, metric=self.metric)

            # Criar o dendrograma usando Matplotlib
            plt.figure(figsize=(10, 7))
            plt.title("Dendrograma")
            dendrogram(Z)
            plt.xlabel("Pontos de Dados")
            plt.ylabel("Distância")
            # Adicionar uma linha horizontal no valor do corte
            plt.axhline(y=cut_value, color='red', linestyle='--')
            self.st.pyplot(plt)
            plt.close()

            # Calcular os clusters com base no valor de corte
            clusters = fcluster(Z, t=cut_value, criterion='distance')

            # Número de clusters
            num_clusters = len(set(clusters))
            self.st.write(f'##### Número de Clusters pelo corte `{cut_value}`:  `{num_clusters}`')
            self.st.write(f'##### Silhouette Score:', self.silhouette_score)
            # Retornar o número de clusters
            return num_clusters
        

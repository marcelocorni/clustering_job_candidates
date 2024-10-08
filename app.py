import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, root_mean_squared_error
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import time
import hashlib
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
import joblib
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Função principal do Streamlit
def main():

    st.set_page_config(page_title='Trabalho 4 - Clusterização', layout='wide')
    

    st.title("Análise de Dados e Agrupamento de Candidatos a Emprego")

    # Função para gerar o URL do Gravatar a partir do e-mail
    def get_gravatar_url(email, size=100):
        email_hash = hashlib.md5(email.strip().lower().encode('utf-8')).hexdigest()
        gravatar_url = f"https://www.gravatar.com/avatar/{email_hash}?s={size}"
        return gravatar_url

    # Definir o e-mail e o tamanho da imagem
    email = "marcelo@desenvolvedor.net"  # Substitua pelo seu e-mail
    size = 200  # Tamanho da imagem

    # Obter o URL do Gravatar
    gravatar_url = get_gravatar_url(email, size)

    # Função para verificar se o valor é numérico e não nulo
    def verificar_tipo_invalido(valor):
        if pd.isna(valor):  # Verifica se o valor é nulo
            return False
        try:
            float(valor)  # Tenta converter para float
            return False
        except ValueError:
            return True

    st.subheader('1. Carregamento dos dados e correção dos nomes das colunas')

    # Carregar os dados
    data = pd.read_csv('data/trab4.csv')

    # Remover espaços em branco no início e no final dos nomes das colunas e converter para Snake Case
    data.columns = (data.columns
                    .str.strip()                   # Remover espaços em branco no início e no final
                    .str.lower()                   # Converter para minúsculas
                    .str.replace(' ', '_')         # Substituir espaços por underscores
                    .str.replace('(', '')          # Remover parênteses
                    .str.replace(')', '')          # Remover parênteses
                    )
    
    # Remover espaços em branco no início e no final dos valores
    data[data.columns] = data[data.columns].applymap(lambda x: x.strip() if isinstance(x, str) else x)

    with st.expander('Código e visualização dos dados', expanded=False):
        with st.popover('Código'):
            st.code('''
                # Carregar os dados
                data = pd.read_csv('data/trab4.csv')

                # Remover espaços em branco no início e no final dos nomes das colunas e converter para Snake Case
                data.columns = (data.columns.str
                                .str.strip()                   # Remover espaços em branco no início e no final
                                .str.lower()                   # Converter para minúsculas
                                .str.replace(' ', '_')         # Substituir espaços por underscores
                                .str.replace('(', '')          # Remover parênteses
                                .str.replace(')', '')          # Remover parênteses
                                )
                    
                    # Remover espaços em branco no início e no final dos valores
                    data[data.columns] = data[data.columns].applymap(lambda x: x.strip() if isinstance(x, str) else x)
                ''')
        
        st.write(data.head(30))
        st.write('Quantidade:', len(data))

    # Remover espaços em branco no início e no final dos nomes das colunas
        

    st.subheader('2. Verificação de valores não-numéricos em colunas que eram esperados valores numéricos')
    with st.expander('Código e visualização dos dados', expanded=False):
        with st.popover('Código'):
            st.code('''
                # Função para verificar se o valor é numérico e não nulo
                def verificar_tipo_invalido(valor):
                    if pd.isna(valor):  # Verifica se o valor é nulo
                        return False
                    try:
                        float(valor)  # Tenta converter para float
                        return False
                    except ValueError:
                        return True

                colunas_numericas = [
                    '10th_percentage', '12th_percentage', 'college_percentage',
                    'english_1', 'english_2', 'english_3', 'english_4',
                    'analytical_skills_1', 'analytical_skills_2', 'analytical_skills_3',
                    'domain_skills_1', 'domain_skills_2', 'domain_test_3','domain_test_4',
                    'quantitative_ability_1', 'quantitative_ability_2', 'quantitative_ability_3', 'quantitative_ability_4'
                ]

                df_tipos_incorretos = data[data[colunas_numericas].applymap(verificar_tipo_invalido).any(axis=1)]
                    
                # Corrigir valores não-numéricos
                data[colunas_numericas] = data[colunas_numericas].apply(pd.to_numeric, errors='coerce')
            ''')

        colunas_numericas = [
            '10th_percentage', '12th_percentage', 'college_percentage',
            'english_1', 'english_2', 'english_3', 'english_4',
            'analytical_skills_1', 'analytical_skills_2', 'analytical_skills_3',
            'domain_skills_1', 'domain_skills_2', 'domain_test_3','domain_test_4',
            'quantitative_ability_1', 'quantitative_ability_2', 'quantitative_ability_3', 'quantitative_ability_4'
        ]

        df_tipos_incorretos = data[data[colunas_numericas].applymap(verificar_tipo_invalido).any(axis=1)]
    
        st.write(df_tipos_incorretos)
        st.write('Quantidade:', len(df_tipos_incorretos))

        data[colunas_numericas] = data[colunas_numericas].apply(pd.to_numeric, errors='coerce')
        
    st.subheader('3. Verificação e tratamento de valores marcados com `MD` e valores nulos')

    with st.expander('Código e visualização dos dados', expanded=False):
        with st.popover('Código'):
            st.code('''
                # Substituir 'MD' por NaN
                data.replace('MD', pd.NA, inplace=True)
            ''')
    
        # Substituir 'MD' por NaN
        data.replace('MD', pd.NA, inplace=True)
    
        # Selecionar os registros onde alguma das colunas tem valor nulo
        df_nulos = data[data.isnull().any(axis=1)]

        # Exibir os registros com valores nulos
        st.write(df_nulos)
        st.write('Quantidade:', len(df_nulos))

    st.subheader('4. Correção de valores ausentes com a média, valores duplicados e remoção das colunas que não serão usadas no agrupamento')

    with st.expander('Código e visualização dos dados', expanded=False):


        def introduce_missing_values(df, missing_rate=0.1):
            """Introduz valores ausentes no DataFrame."""
            df = df.copy()
            n_missing = int(np.floor(missing_rate * df.size))
            missing_indices = np.random.choice(df.size, n_missing, replace=False)
            df.values.ravel()[missing_indices] = np.nan
            return df

        def find_best_n_neighbors(data: pd.DataFrame, neighbors_list: list = [3, 5, 7, 10], missing_rate=0.1) -> int:
            """
            Encontra o melhor número de vizinhos para KNNImputer com base na avaliação manual.

            Parameters:
            - data (pd.DataFrame): O DataFrame contendo dados numéricos com valores ausentes.
            - neighbors_list (list): Lista de valores para n_neighbors a serem testados.
            - missing_rate (float): Taxa de valores ausentes a serem introduzidos no conjunto de teste para avaliação.

            Returns:
            - int: O melhor número de vizinhos.
            """
            
            # Selecionar colunas numéricas
            num_columns = data.select_dtypes(include=['float64', 'int64']).columns
            X = data[num_columns]
            
            # Dividir dados em treino e teste
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            
            # Introduzir valores ausentes no conjunto de teste
            X_test_missing = introduce_missing_values(X_test, missing_rate)
            
            best_score = float('inf')
            best_n_neighbors = neighbors_list[0]
            
            # Testar diferentes valores de n_neighbors
            for n in neighbors_list:
                imputer = KNNImputer(n_neighbors=n)
                
                # Imputar os dados de treinamento e teste
                X_train_imputed = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test_missing)
                
                # Avaliar a imputação comparando com dados reais (dados de teste sem ausências)
                score = root_mean_squared_error(X_test, X_test_imputed)
                
                if score < best_score:
                    best_score = score
                    best_n_neighbors = n

            return best_n_neighbors,num_columns

        
        
        imputer = KNNImputer(n_neighbors=5)
        data_clone = data.copy()
        num_columns = data_clone.select_dtypes(include=['number']).columns
        data_clone[num_columns] = imputer.fit_transform(data_clone[num_columns])
        # Imputar valores ausentes usando o KNNImputer, considerando apenas colunas numéricas
        best_n_neighbors,num_columns = find_best_n_neighbors(data_clone)

        imputer = KNNImputer(n_neighbors=best_n_neighbors)
        data[num_columns] = imputer.fit_transform(data[num_columns])
        data['performance'].fillna(data['performance'].mode()[0], inplace=True)
        
        
        # Remover duplicados
        st.write('Quantidade de duplicados:', data.duplicated().sum())
        data.drop_duplicates(inplace=True)

        # Remover colunas que não serão usadas no agrupamento
        data_clustering = data.drop(columns=['performance', 'candidate_id', 'name','number_of_characters_in_original_name','state_location'])

        with st.popover('Código'):
            st.code('''
                # Imputar valores ausentes com a média, considerando apenas colunas numéricas
                num_columns = data.select_dtypes(include=['number']).columns
                data[num_columns] = data[num_columns].fillna(data[num_columns].mean())

                # Remover colunas que não serão usadas no agrupamento
                data_clustering = data.drop(columns=['performance', 'candidate_id', 'name','number_of_characters_in_original_name','state_location'])
                    ''')
        
        st.write(data_clustering.head(30))
        st.write('Quantidade:', len(data_clustering))

    st.subheader('5. Mapeamentos e transformações necessárias para a clusterização')

    with st.expander('Código e visualização dos dados', expanded=False):

        # Função para converter 'Y7' para '2007', 'Y20' para '2020', etc.
        def convert_year(year_str):
            year_number = int(year_str[1:])  # Remove o 'Y' e converte o restante para número
            return 2000 + year_number  # Adiciona 2000 para obter o ano completo

        # Converter 'Year of Birth', '10th Completion Year', '12th Completion Year' e 'College Completion Year' para números
        data_clustering['year_of_birth'] = data_clustering['year_of_birth'].map(convert_year).astype('int')
        data_clustering['10th_completion_year'] = data_clustering['10th_completion_year'].map(convert_year).astype('int')
        data_clustering['12th_completion_year'] = data_clustering['12th_completion_year'].map(convert_year).astype('int')
        data_clustering['year_of_completion_of_college'] = data_clustering['year_of_completion_of_college'].map(convert_year).astype('int')

        # Mapeamento dos meses para números
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }

        # Converter 'Month of Birth' para números
        data_clustering['month_of_birth'] = data_clustering['month_of_birth'].map(month_map)

        # Calcular a data de nascimento aproximada
        data_clustering['birth_date'] = pd.to_datetime(data_clustering['year_of_birth'].astype(str) + '-' + data_clustering['month_of_birth'].astype(str) + '-01')

        # Calcular a idade aproximada
        today = datetime.today()
        aproximated_age = data_clustering['birth_date'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))
        # Inserir a idade aproximada no DataFrame na primeira posição
        data_clustering.insert(0, 'aproximated_age', aproximated_age)
        data.insert(0, 'aproximated_age', aproximated_age)

        # Remover colunas que não serão usadas no agrupamento e as colunas que não fazem sentido
        data_clustering = data_clustering.drop(columns=['birth_date', 'month_of_birth', 'year_of_birth','year_of_completion_of_college','10th_completion_year','12th_completion_year','year_of_completion_of_college'])

        data_clustering = pd.get_dummies(data_clustering, columns=['degree_of_study', 'specialization_in_study'], drop_first=True)
        
        st.write(data_clustering.head(30))
        st.write("DUMMIES")

        # Mapeamento do gênero para números
        gender_map = {
            'A': 1, 'B': 2
        }

        # Converter 'gender' para números
        data_clustering['gender'] = data_clustering['gender'].map(gender_map)

        with st.popover('Código'):
            st.code('''
                # Função para converter 'Y7' para '2007', 'Y20' para '2020', etc.
                def convert_year(year_str):
                    year_number = int(year_str[1:])  # Remove o 'Y' e converte o restante para número
                    return 2000 + year_number  # Adiciona 2000 para obter o ano completo

                # Converter 'Year of Birth', '10th Completion Year', '12th Completion Year' e 'College Completion Year' para números
                data_clustering['year_of_birth'] = data_clustering['year_of_birth'].map(convert_year).astype('int')
                data_clustering['10th_completion_year'] = data_clustering['10th_completion_year'].map(convert_year).astype('int')
                data_clustering['12th_completion_year'] = data_clustering['12th_completion_year'].map(convert_year).astype('int')
                data_clustering['year_of_completion_of_college'] = data_clustering['year_of_completion_of_college'].map(convert_year).astype('int')

                # Mapeamento dos meses para números
                month_map = {
                    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                }

                # Converter 'Month of Birth' para números
                data_clustering['month_of_birth'] = data_clustering['month_of_birth'].map(month_map)

                # Calcular a data de nascimento aproximada
                data_clustering['birth_date'] = pd.to_datetime(data_clustering['year_of_birth'].astype(str) + '-' + data_clustering['month_of_birth'].astype(str) + '-01')

                # Calcular a idade aproximada
                today = datetime.today()
                aproximated_age = data_clustering['birth_date'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))
                # Inserir a idade aproximada no DataFrame na primeira posição
                data_clustering.insert(0, 'aproximated_age', aproximated_age)

                # Remover colunas que não serão usadas no agrupamento e as colunas que não fazem sentido
                data_clustering = data_clustering.drop(columns=['birth_date', 'month_of_birth', 'year_of_birth','degree_of_study','specialization_in_study','year_of_completion_of_college','10th_completion_year','12th_completion_year','year_of_completion_of_college'])

                # Mapeamento do gênero para números
                gender_map = {
                    'A': 1, 'B': 2
                }

                # Converter 'gender' para números
                data_clustering['gender'] = data_clustering['gender'].map(gender_map)
            ''')

        st.write(data_clustering.head(30))
        st.write('Quantidade:', len(data_clustering))

    st.subheader('6. Agrupamento de atributos para redução da dimensionalidade')
    with st.expander('Código e visualização dos dados', expanded=False):
        
        # # Criar novas colunas com a média dos grupos
        # data_clustering['english'] = data_clustering[['english_1', 'english_2', 'english_3', 'english_4']].mean(axis=1)
        # data_clustering['analytical_skills'] = data_clustering[['analytical_skills_1', 'analytical_skills_2', 'analytical_skills_3']].mean(axis=1)
        # data_clustering['domain_skills'] = data_clustering[['domain_skills_1', 'domain_skills_2', 'domain_test_3', 'domain_test_4']].mean(axis=1)
        # data_clustering['quantitative_ability'] = data_clustering[['quantitative_ability_1', 'quantitative_ability_2', 'quantitative_ability_3', 'quantitative_ability_4']].mean(axis=1)
        # data_clustering['college'] = data_clustering[['college_percentage','10th_percentage','12th_percentage']].mean(axis=1)


        # # Remover as colunas antigas, se necessário
        # data_clustering = data_clustering.drop(columns=['english_1', 'english_2', 'english_3', 'english_4',
        #                         'analytical_skills_1', 'analytical_skills_2', 'analytical_skills_3',
        #                         'domain_skills_1', 'domain_skills_2', 'domain_test_3', 'domain_test_4',
        #                         'quantitative_ability_1', 'quantitative_ability_2', 'quantitative_ability_3', 'quantitative_ability_4',
        #                         'college_percentage','10th_percentage','12th_percentage'])
        
        with st.popover('Código'):
            st.code('''
                # Criar novas colunas com a média dos grupos
                data_clustering['english'] = data_clustering[['english_1', 'english_2', 'english_3', 'english_4']].mean(axis=1)
                data_clustering['analytical_skills'] = data_clustering[['analytical_skills_1', 'analytical_skills_2', 'analytical_skills_3']].mean(axis=1)
                data_clustering['domain_skills'] = data_clustering[['domain_skills_1', 'domain_skills_2', 'domain_test_3', 'domain_test_4']].mean(axis=1)
                data_clustering['quantitative_ability'] = data_clustering[['quantitative_ability_1', 'quantitative_ability_2', 'quantitative_ability_3', 'quantitative_ability_4']].mean(axis=1)
                data_clustering['college'] = data_clustering[['college_percentage','10th_percentage','12th_percentage']].mean(axis=1)


                # Remover as colunas antigas, se necessário
                data_clustering = data_clustering.drop(columns=['english_1', 'english_2', 'english_3', 'english_4',
                                        'analytical_skills_1', 'analytical_skills_2', 'analytical_skills_3',
                                        'domain_skills_1', 'domain_skills_2', 'domain_test_3', 'domain_test_4',
                                        'quantitative_ability_1', 'quantitative_ability_2', 'quantitative_ability_3', 'quantitative_ability_4',
                                        'college_percentage','10th_percentage','12th_percentage'])
            ''')

        st.write(data_clustering.head(30))
        st.write('Quantidade:', len(data_clustering))

    st.subheader('`7.` Análise de Correlação e Distribuição dos dados categóricos (aproximated_age, gender)')
    with st.expander('Código e visualização dos dados', expanded=False):

        def calculate_vif(dataf: pd.DataFrame, threshold: float = 10.0):
            """
            Calcula o VIF para cada variável e lista as variáveis com VIF acima do limiar.

            Parameters:
            - data (pd.DataFrame): O DataFrame contendo os dados numéricos.
            - threshold (float): O limiar para VIF considerado alto. O padrão é 10.0.

            Returns:
            - pd.DataFrame: Um DataFrame com as variáveis que têm VIF acima do limiar.
            """
            # Selecionar colunas numéricas
            num_columns = dataf.select_dtypes(include=[np.number]).columns

            # Adicionar constante para o cálculo do VIF
            X = sm.add_constant(dataf[num_columns])
            
            # Calcular VIF para cada variável
            vif_data = pd.DataFrame()
            vif_data['Feature'] = X.columns
            vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            
            # Filtrar as variáveis com VIF alto
            vif_high = vif_data[vif_data['VIF'] > threshold]
            #vif_high = vif_high.drop(0)  # Remover a constante
            vif_high = vif_high.sort_values(by='VIF', ascending=False)
            
            return vif_high

        with st.popover('Código'):
            st.code('''
                # Exibir a matriz de correlação usando Plotly
                fig = px.imshow(corr_matrix,
                                labels=dict(x="Atributos", y="Atributos", color="Correlação"),
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1)

                fig.update_layout(title='Matriz de Correlação',
                                width=800, height=800,
                                margin=dict(l=50, r=50, t=50, b=50))

                st.plotly_chart(fig, use_container_width=True)
                    
                # Calcular a matriz de correlação
                num_columns = data_clustering.select_dtypes(include=['number']).columns
                corr_matrix = data_clustering[num_columns].corr()
                # Identificar atributos com alta correlação
                corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort").drop_duplicates()
                high_corr = corr_pairs[(corr_pairs > threshold) & (corr_pairs < 1)]

                st.write("Pares de atributos com alta correlação:")
                st.write(high_corr)
            ''')



        with st.sidebar.expander("`[7]` Config. de Correlação"):
            # Identificar atributos com alta correlação
            threshold = st.number_input('Threshold de Correlação', -1.0, 1.0, 0.70, 0.01)
            # Excluir coluna com alta correlação
            colunas_para_excluir = st.multiselect('Coluna para excluir', options=data_clustering.columns,default=['specialization_in_study_F','specialization_in_study_H','specialization_in_study_K','domain_skills_1','degree_of_study_X','degree_of_study_Y','degree_of_study_Z'])

        # Calcular a matriz de correlação
        num_columns = data_clustering.select_dtypes(include=['number']).columns
        corr_matrix = data_clustering[num_columns].corr()
        # Identificar atributos com alta correlação
        corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort").drop_duplicates()
        high_corr = corr_pairs[(corr_pairs > threshold) & (corr_pairs < 1)]

        st.write("Correlações:")
        st.write(high_corr)

        st.write("Fator de Inflação da Variância (VIF):")
        # Calcular VIF e listar variáveis com VIF alto
        vif_high = calculate_vif(data_clustering, threshold=5.0)
        st.write(vif_high)

        # Exibir a matriz de correlação usando Plotly
        fig = px.imshow(corr_matrix,
                        labels=dict(x="Atributos", y="Atributos", color="Correlação"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1)

        fig.update_layout(title='Matriz de Correlação',
                        width=800, height=800,
                        margin=dict(l=50, r=50, t=50, b=50))

        st.plotly_chart(fig, use_container_width=True)

        with st.popover('Código'):
            st.code('''
                # Exibir gráfico de distribuição dos dados por idade aproximada
                fig_age = px.histogram(data_clustering, x=data_clustering['aproximated_age'], title='Distribuição por Idade Aproximada')
                st.plotly_chart(fig_age)

                # Exibir gráfico de distribuição dos dados por gênero
                fig_gender = px.histogram(data_clustering, x=data_clustering['gender'], title='Distribuição por Gênero')
                st.plotly_chart(fig_gender)
            ''')

        # Exibir gráfico de distribuição dos dados por idade aproximada
        fig_age = px.histogram(data_clustering, x=data_clustering['aproximated_age'], title='Distribuição por Idade Aproximada')
        st.plotly_chart(fig_age)

        # Exibir gráfico de distribuição dos dados por gênero
        fig_gender = px.histogram(data_clustering, x=data_clustering['gender'], title='Distribuição por Gênero')
        st.plotly_chart(fig_gender)

        if colunas_para_excluir:
            data_clustering.drop(columns=colunas_para_excluir, inplace=True)

    st.subheader('8. Remoção de atributos com baixa correlação ou nenhum correlação')
    with st.expander('Código e visualização dos dados', expanded=False):
        with st.popover('Código'):
            st.code('''
                # Remover colunas com baixa correlação
                data_clustering = data_clustering.drop(columns=['aproximated_age', 'gender'])
            ''')

        data_clustering = data_clustering.drop(columns=['aproximated_age', 'gender'])

        st.write(data_clustering.head(30))
        st.write('Quantidade:', len(data_clustering))

    
    st.subheader('`9.` Identificação/Remoção de Outliers')

    with st.expander('Código e visualização dos dados', expanded=False):

        def show_outliers(title):
            # Criar um boxplot para visualizar os outliers
            fig = px.box(data_clustering, points="all", title=title)
            fig.update_layout(xaxis_tickangle=-90)  # Girar os nomes das colunas para melhor visualização
            st.plotly_chart(fig)

        # Aplicar logaritmo para normalizar os dados
        data_clustering = np.log1p(data_clustering)
        
        show_outliers('Boxplot dos Dados Antes da Remoção de Outliers')

        # Calcular o Z-Score para cada valor em cada coluna
        z_scores = np.abs(stats.zscore(data_clustering))

        with st.sidebar.expander("`[9]` Config. do Z-Score"):
            # Configurar o treshold do Z-Score
            threshold = st.number_input('Threshold de Z-Score', 0.000, 10.000, 2.350, 0.001, format="%.3f")

        numero_outliers = data_clustering[(z_scores > threshold).any(axis=1)].shape[0]
        st.write(f"Quantidade de outliers(`{numero_outliers}`) com Z-Score maior que `{threshold}`")

        # Identificar os índices dos registros que são outliers
        outlier_indices = np.where((z_scores > threshold).any(axis=1))[0]

        # Remover os outliers dos dois DataFrames
        existing_indices_data = data.index.intersection(outlier_indices)
        if not existing_indices_data.empty:
            data = data.drop(existing_indices_data, axis=0).reset_index(drop=True)

        existing_indices_data_clustering = data_clustering.index.intersection(outlier_indices)
        if not existing_indices_data_clustering.empty:
            data_clustering = data_clustering.drop(existing_indices_data_clustering, axis=0).reset_index(drop=True)
            
        # Outliers após a remoção
        show_outliers('Boxplot dos Dados Após a Remoção dos Outliers')

        st.dataframe(data_clustering.head(30))
        st.write('Quantidade:', len(data_clustering))

    st.subheader('`10.` Aplicação do K-means')
    with st.expander('Código e visualização dos dados', expanded=False):

        with st.popover('Código'):
            st.code('''
                    
                # Gráfico de Pareto da Variância Explicada
                st.header("Gráfico de Pareto da Variância Explicada")
                fig_pareto = px.bar(
                    x=[f'PC{i+1}' for i in range(n_components_pca)],
                    y=explained_variance,
                    labels={'x': 'Componentes Principais', 'y': 'Proporção da Variância Explicada'},
                    title='Variância Explicada pelos Componentes Principais'
                )
                st.plotly_chart(fig_pareto)
                    
                # Comparação da variância
                st.header("Comparação da Variância")
                raw_variance_before_scaling = np.var(data_clustering, axis=0).mean()
                st.write("Variância Média dos Dados Brutos Antes da Normalização:", raw_variance_before_scaling)
                raw_variance_after_scaling  = np.var(data_scaled, axis=0).mean()
                st.write("Variância Média dos Dados Brutos Após a Normalização:", raw_variance_after_scaling )
                st.write("Proporção da Variância Explicada pelos Componentes Principais do PCA:", explained_variance)
            ''')

        # Função para gerar o gráfico do cotovelo
        def plot_cotovelo(data_scaled):
            inertia = []
            range_n_clusters = range(1, 11)  # Definindo o intervalo de clusters

            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(data_scaled)
                inertia.append(kmeans.inertia_)

            # Criando o gráfico do cotovelo
            fig = px.line(x=list(range_n_clusters), y=inertia, markers=True,
                        labels={'x': 'Número de Clusters', 'y': 'Inércia'},
                        title='Método do Cotovelo')
            return fig
        
        def plot_dispersao_kmeans(data, data_scaled, n_clusters, n_components=2):
            # Aplicar PCA para reduzir a dimensionalidade para o número de componentes especificado
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(data_scaled)
            
            if n_components == 2:
                data_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                data_pca['cluster_kmeans'] = data['cluster_kmeans']
                
                # Criar o gráfico de dispersão 2D
                fig = px.scatter(data_pca, x='PC1', y='PC2', color='cluster_kmeans', 
                                title=f'Dispersão dos Clusters (n_clusters={n_clusters})',
                                labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'})
            
            elif n_components == 3:
                data_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
                data_pca['cluster_kmeans'] = data['cluster_kmeans']
                
                # Criar o gráfico de dispersão 3D
                fig = px.scatter_3d(data_pca, x='PC1', y='PC2', z='PC3', color='cluster_kmeans',
                                    title=f'Dispersão 3D dos Clusters (n_clusters={n_clusters})',
                                    labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2', 'PC3': 'Componente Principal 3'})
            
            else:
                raise ValueError("n_components deve ser 2 ou 3 para a visualização.")
            
            return fig


        # Transformar os dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_clustering)

        # Sidebar para configuração do PCA e K-means
        with st.sidebar.expander("`[10-11]` Config. PCA"):
            n_components_pca = st.radio('Nº de Componentes Principais - PCA', [2, 3])

        # Sidebar para configuração do PCA e K-means
        with st.sidebar.expander("`[10]` Config. K-means"):
            n_clusters = st.number_input('Nº de Clusters - K-means', 2, 20, 2)
            random_state = st.number_input('Random State - K-means', 0, 1000, 43)

        # Aplicar PCA para reduzir a dimensionalidade a n_clusters componentes principais
        pca = PCA(n_components=n_components_pca)
        data_pca = pca.fit_transform(data_scaled)
        explained_variance = pca.explained_variance_ratio_
            
        # Gráfico de Pareto da Variância Explicada
        st.header("Gráfico de Pareto da Variância Explicada")
        fig_pareto = px.bar(
            x=[f'PC{i+1}' for i in range(n_components_pca)],
            y=explained_variance,
            labels={'x': 'Componentes Principais', 'y': 'Proporção da Variância Explicada'},
            title='Variância Explicada pelos Componentes Principais'
        )
        st.plotly_chart(fig_pareto)

        # Comparação da variância
        st.header("Comparação da Variância")
        raw_variance_before_scaling = np.var(data_clustering, axis=0).mean()
        st.write("Variância Média dos Dados Brutos Antes da Normalização:", raw_variance_before_scaling)
        raw_variance_after_scaling  = np.var(data_scaled, axis=0).mean()
        st.write("Variância Média dos Dados Brutos Após a Normalização:", raw_variance_after_scaling )
        st.write("Proporção da Variância Explicada pelos Componentes Principais do PCA:", explained_variance)

        with st.popover('Código'):
            st.code('''
                # Função para gerar o gráfico do cotovelo
                def plot_cotovelo(data_scaled):
                    inertia = []
                    range_n_clusters = range(1, 11)  # Definindo o intervalo de clusters

                    for n_clusters in range_n_clusters:
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                        kmeans.fit(data_scaled)
                        inertia.append(kmeans.inertia_)

                    # Criando o gráfico do cotovelo
                    fig = px.line(x=list(range_n_clusters), y=inertia, markers=True,
                                labels={'x': 'Número de Clusters', 'y': 'Inércia'},
                                title='Método do Cotovelo')
                    return fig
            ''')
        # Exibir gráfico do cotovelo
        st.header('Método do Cotovelo')
        fig_cotovelo = plot_cotovelo(data_pca)
        st.plotly_chart(fig_cotovelo)
        
        # Aplicar k-means com n_clusters clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        data['cluster_kmeans'] = kmeans.fit_predict(data_pca)
        
        kmeans_silhouette = silhouette_score(data_pca, data['cluster_kmeans'])
        st.write("K-Means Silhouette Score:", kmeans_silhouette)

        with st.popover('Código'):
            st.code('''
                # Função para gerar o gráfico de dispersão
                def plot_dispersao_kmeans(data, data_scaled, n_clusters):
                    # Aplicar PCA para reduzir a dimensionalidade a 2 componentes principais
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(data_scaled)
                    data_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                    data_pca['cluster_kmeans'] = data['cluster_kmeans']

                    # Criar o gráfico de dispersão
                    fig = px.scatter(data_pca, x='PC1', y='PC2', color='cluster_kmeans', 
                                    title=f'Dispersão dos Clusters (n_clusters={n_clusters})',
                                    labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'})
                    return fig
            ''')

        # Exibir gráfico de dispersão dos clusters
        st.header('Gráfico de Dispersão dos Clusters')
        scatter_fig = plot_dispersao_kmeans(data, data_scaled, n_clusters, n_components_pca)
        st.plotly_chart(scatter_fig)

        # Exibir a distribuição dos clusters
        cluster_distribution = data['cluster_kmeans'].value_counts()

        with st.popover('Código'):
            st.code('''
                # Exibir gráfico de distribuição dos clusters
                st.header('Distribuição dos Clusters')
                fig = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values,
                            labels={'x': 'Cluster', 'y': 'Número de Instâncias'},
                            title='Distribuição dos Clusters')
                st.plotly_chart(fig)
            ''')

        # Exibir gráfico de distribuição dos clusters
        st.header('Distribuição dos Clusters')
        fig_kmenas_distribuicao = px.bar(cluster_distribution, x=cluster_distribution.index, y=cluster_distribution.values,
                    labels={'x': 'Cluster', 'y': 'Número de Instâncias'},
                    title='Distribuição dos Clusters')
        st.plotly_chart(fig_kmenas_distribuicao)

        # Analisar a distribuição da variável "Performance" dentro de cada cluster
        performance_distribution = data.groupby('cluster_kmeans')['performance'].value_counts().unstack().fillna(0)
        aproximated_age_distribution = data.groupby('cluster_kmeans')['aproximated_age'].value_counts().unstack().fillna(0)
        gender_distribution = data.groupby('cluster_kmeans')['gender'].value_counts().unstack().fillna(0)

        with st.popover('Código'):
            st.code('''
                # Exibir gráfico de distribuição de Idade Aproximada por Cluster
                st.header('Distribuição da Idade Aproximada por Cluster')
                fig_age = px.bar(aproximated_age_distribution, barmode='group',
                                        labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                        title='Distribuição da Idade Aproximada por Cluster')
                st.plotly_chart(fig_age)
            ''')
            
        # Exibir gráfico de distribuição de Idade Aproximada por Cluster
        st.header('Distribuição da Idade Aproximada por Cluster')
        fig_age = px.bar(aproximated_age_distribution, barmode='group',
                                labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                title='Distribuição da Idade Aproximada por Cluster')
        st.plotly_chart(fig_age)

        with st.popover('Código'):
            st.code('''
                # Exibir gráfico de distribuição do Gênero por Cluster
                st.header('Distribuição do Gênero por Cluster')
                fig_gender = px.bar(gender_distribution, barmode='group',
                                        labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                        title='Distribuição do Gênero por Cluster')
                st.plotly_chart(fig_gender)
            ''')

        # Exibir gráfico de distribuição do Gênero por Cluster
        st.header('Distribuição do Gênero por Cluster')
        fig_gender = px.bar(gender_distribution, barmode='group',
                                labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                title='Distribuição do Gênero por Cluster')
        st.plotly_chart(fig_gender)

        with st.popover('Código'):
            st.code('''
                # Exibir gráfico de distribuição de Performance por Cluster
                st.header('Distribuição da Performance por Cluster')
                fig_performance = px.bar(performance_distribution, barmode='group',
                                        labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                        title='Distribuição da Performance por Cluster')
                st.plotly_chart(fig_performance)
            ''')
            
        # Exibir gráfico de distribuição de Performance por Cluster
        st.header('Distribuição da Performance por Cluster')
        fig_performance = px.bar(performance_distribution, barmode='group',
                                labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                title='Distribuição da Performance por Cluster')
        st.plotly_chart(fig_performance)

        
    st.subheader('`11`. Aplicação do Agglomerative Clustering')
    with st.expander('Código e visualização dos dados', expanded=False):
        
        # Função para aplicar Agglomerative Clustering
        def apply_agglomerative_clustering(data_scaled, n_clusters, linkage, metric):
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric, compute_distances=False, compute_full_tree='auto')
            labels = agglomerative.fit_predict(data_scaled)
            return labels

        def plot_dispersao_aglomerative(data, data_scaled, n_clusters, n_components=2):
            # Aplicar PCA para reduzir a dimensionalidade para o número de componentes especificado
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(data_scaled)
            
            if n_components == 2:
                data_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
                data_pca['cluster_agglomerative'] = data['cluster_agglomerative']
                
                # Criar o gráfico de dispersão 2D
                fig = px.scatter(data_pca, x='PC1', y='PC2', color='cluster_agglomerative', 
                                title=f'Dispersão dos Clusters (n_clusters={n_clusters}) - Agglomerative Clustering',
                                labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'})
            
            elif n_components == 3:
                data_pca = pd.DataFrame(data=components, columns=['PC1', 'PC2', 'PC3'])
                data_pca['cluster_agglomerative'] = data['cluster_agglomerative']
                
                # Criar o gráfico de dispersão 3D
                fig = px.scatter_3d(data_pca, x='PC1', y='PC2', z='PC3', color='cluster_agglomerative',
                                    title=f'Dispersão 3D dos Clusters (n_clusters={n_clusters}) - Agglomerative Clustering',
                                    labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2', 'PC3': 'Componente Principal 3'})
            
            else:
                raise ValueError("n_components deve ser 2 ou 3 para a visualização.")
            
            return fig

        with st.popover('Código'):
            st.code('''
                # Função para aplicar Agglomerative Clustering
                def apply_agglomerative_clustering(data_scaled, n_clusters, linkage, metric):
                    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
                    labels = agglomerative.fit_predict(data_scaled)
                    return labels
            ''')

        # Sidebar para controlar o valor do corte
        with st.sidebar.expander("`[11]` Config. Dendrograma"):
            cut_value = st.number_input("Valor de Corte (Distância)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)

        # Configurações na Sidebar para o Agglomerative Clustering
        with st.sidebar.expander("`[11]` Config. Agglomerative"):
            n_clusters_agglomerative = st.number_input('Nº de Clusters - Agglomerative', 2, 200, 2)
            linkage_method = st.selectbox('Método de Ligação', ['ward','average', 'complete', 'single'])
            metric_method = st.selectbox('Métrica', ['euclidean','cosine'])


            # Verifique se a métrica é compatível com o método de ligação escolhido
            if linkage_method == 'ward' and metric_method != 'euclidean':
                st.warning("O método de ligação 'ward' só funciona com a métrica 'euclidean'. Ajustando para 'euclidean'.")
                metric_method = 'euclidean'
        
        # Aplicar Agglomerative Clustering
        agglomerative_labels = apply_agglomerative_clustering(data_scaled, n_clusters_agglomerative, linkage_method, metric_method)
        data['cluster_agglomerative'] = agglomerative_labels
        
        # Exibir o dendrograma e o número de clusters e o silhouette score
        from lib.corni.agrupamento import corni_agglomerative_clustering
        ag = corni_agglomerative_clustering(st=st,data=data_scaled, n_clusters=n_clusters_agglomerative, metric= metric_method, linkage=linkage_method)
        ag.apply()
        n = ag.plot_dendrogram_matplotlib(cut_value=cut_value)

        # Exibir gráfico de dispersão dos clusters
        st.header('Gráfico de Dispersão dos Clusters - Agglomerative Clustering')
        scatter_fig_agglomerative = plot_dispersao_aglomerative(data, data_scaled, n_clusters_agglomerative, n_components_pca)
        st.plotly_chart(scatter_fig_agglomerative)

        # Exibir a distribuição dos clusters
        cluster_distribution_agglomerative = data['cluster_agglomerative'].value_counts()


        st.header('Distribuição dos Clusters - Agglomerative Clustering')
        fig_agglomerative_distribution = px.bar(cluster_distribution_agglomerative, x=cluster_distribution_agglomerative.index, y=cluster_distribution_agglomerative.values,
                        labels={'x': 'Cluster', 'y': 'Número de Instâncias'},
                        title='Distribuição dos Clusters - Agglomerative Clustering')
        st.plotly_chart(fig_agglomerative_distribution)
        
        # Análise adicional da distribuição das variáveis em cada cluster
        performance_distribution_agglomerative = data.groupby('cluster_agglomerative')['performance'].value_counts().unstack().fillna(0)
        aproximated_age_distribution_agglomerative = data.groupby('cluster_agglomerative')['aproximated_age'].value_counts().unstack().fillna(0)
        gender_distribution_agglomerative = data.groupby('cluster_agglomerative')['gender'].value_counts().unstack().fillna(0)
        
        st.header('Distribuição da Idade Aproximada por Cluster - Agglomerative Clustering')
        fig_age_agglomerative = px.bar(aproximated_age_distribution_agglomerative, barmode='group',
                                        labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                        title='Distribuição da Idade Aproximada por Cluster - Agglomerative Clustering')
        st.plotly_chart(fig_age_agglomerative)

        st.header('Distribuição do Gênero por Cluster - Agglomerative Clustering')
        fig_gender_agglomerative = px.bar(gender_distribution_agglomerative, barmode='group',
                                        labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                        title='Distribuição do Gênero por Cluster - Agglomerative Clustering')
        st.plotly_chart(fig_gender_agglomerative)

        st.header('Distribuição da Performance por Cluster - Agglomerative Clustering')
        fig_performance_agglomerative = px.bar(performance_distribution_agglomerative, barmode='group',
                                            labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                            title='Distribuição da Performance por Cluster - Agglomerative Clustering')
        st.plotly_chart(fig_performance_agglomerative)

    st.subheader('`12.` Aplicação do DBSCAN')
    with st.expander('Código e visualização dos dados', expanded=False):

        with st.sidebar.expander("`[12]` Config. DBSCAN"):
            eps = st.number_input('EPS - DBSCAN', min_value=0.001, max_value=100.000, value=8.000, step=0.001, format="%.3f")
            min_samples = st.number_input('Min Samples - DBSCAN', min_value=1, max_value=1000, value=30, step=1)

        def plot_knn_distance(data_scaled, k):
            # Ajustar o modelo de vizinhos mais próximos
            nearest_neighbors = NearestNeighbors(n_neighbors=k)
            neighbors = nearest_neighbors.fit(data_scaled)

            # Distâncias para o k-ésimo vizinho mais próximo
            distances, indices = neighbors.kneighbors(data_scaled)

            # Ordenar as distâncias para o k-ésimo vizinho mais próximo
            distances = np.sort(distances[:, k-1])

            # Criar o gráfico usando Plotly
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                y=distances,
                mode='lines',
                name=f'{k}ª distância mais próxima'
            ))

            fig.update_layout(
                title='Gráfico de Distância dos Vizinhos mais Próximos',
                xaxis_title='Pontos ordenados',
                yaxis_title=f'{k}ª distância mais próxima',
                height=600
            )

            return fig
        
        def plot_dbscan_density(data, labels):
            # Adicionar rótulos ao DataFrame
            data_df = pd.DataFrame(data, columns=['X', 'Y'])
            data_df['cluster'] = labels

            # Separar os clusters do ruído
            clustered_data = data_df[data_df['cluster'] != -1]
            noise_data = data_df[data_df['cluster'] == -1]

            # Criar gráfico de densidade para os clusters
            fig = ff.create_2d_density(
                clustered_data['X'], clustered_data['Y'],
                colorscale='Viridis',
                hist_color='rgb(0, 0, 100)',
                point_size=3,
                title="Density Plot dos Clusters - DBSCAN"
            )
            
            # Adicionar os pontos de ruído ao gráfico
            fig.add_trace(
                go.Scatter(
                    x=noise_data['X'],
                    y=noise_data['Y'],
                    mode='markers',
                    marker=dict(color='red', size=5),
                    name='Ruído'
                )
            )

            return fig

        def plot_dbscan_results(data, labels, n_components=2):
            # Adicionar rótulos ao DataFrame
            data_pca = pd.DataFrame(data, columns=[f'PC{i+1}' for i in range(n_components)])
            data_pca['cluster'] = labels
            
            # Diferenciar os ruídos dos clusters
            data_pca['cluster_label'] = data_pca['cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')
            
            if n_components == 2:
                # Criar o gráfico de dispersão 2D
                fig = px.scatter(data_pca, x='PC1', y='PC2', color='cluster_label',
                                title='Dispersão dos Clusters - DBSCAN',
                                labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2'})
            elif n_components == 3:
                # Criar o gráfico de dispersão 3D
                fig = px.scatter_3d(data_pca, x='PC1', y='PC2', z='PC3', color='cluster_label',
                                    title='Dispersão 3D dos Clusters - DBSCAN',
                                    labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2', 'PC3': 'Componente Principal 3'})
            else:
                raise ValueError("n_components deve ser 2 ou 3 para a visualização.")
            
            return fig

        def apply_dbscan(data_scaled, eps, min_samples):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data_scaled)
            return labels

        # Aplicar o DBSCAN
        dbscan_labels = apply_dbscan(data_scaled, eps, min_samples)
        data['cluster_dbscan'] = dbscan_labels

        # Configurações do DBSCAN
        k = min_samples  # ou um valor definido pelo usuário

        # Exibir o gráfico no KNN
        fig_knn_distance = plot_knn_distance(data_scaled, k)
        st.plotly_chart(fig_knn_distance)

        # Exibir o gráfico de densidade dos clusters
        fig_density = plot_dbscan_density(data_pca, dbscan_labels)
        st.plotly_chart(fig_density, use_container_width=True)

        dbscan_silhouette = silhouette_score(data_pca, data['cluster_dbscan'])
        st.write("DBSCAN Silhouette Score:", dbscan_silhouette)

        # Exibir o gráfico dos resultados do DBSCAN
        fig_dbscan = plot_dbscan_results(data_pca, dbscan_labels, n_components_pca)
        st.plotly_chart(fig_dbscan)

        # Exibir a distribuição dos clusters
        cluster_distribution_dbscan = data['cluster_dbscan'].value_counts()

        st.header('Distribuição dos Clusters - DBSCAN')
        fig_dbscan_distribution = px.bar(cluster_distribution_dbscan, x=cluster_distribution_dbscan.index, y=cluster_distribution_dbscan.values,
                        labels={'x': 'Cluster', 'y': 'Número de Instâncias'},
                        title='Distribuição dos Clusters - DBSCAN')
        st.plotly_chart(fig_dbscan_distribution)
        
        # Análise adicional da distribuição das variáveis em cada cluster
        performance_distribution_dbscan = data.groupby('cluster_dbscan')['performance'].value_counts().unstack().fillna(0)
        aproximated_age_distribution_dbscan = data.groupby('cluster_dbscan')['aproximated_age'].value_counts().unstack().fillna(0)
        gender_distribution_dbscan = data.groupby('cluster_dbscan')['gender'].value_counts().unstack().fillna(0)
        
        st.header('Distribuição da Idade Aproximada por Cluster - DBSCAN')
        fig_age_dbscan = px.bar(aproximated_age_distribution_dbscan, barmode='group',
                                        labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                        title='Distribuição da Idade Aproximada por Cluster - DBSCAN')
        st.plotly_chart(fig_age_dbscan)

        st.header('Distribuição do Gênero por Cluster - DBSCAN')
        fig_gender_dbscan = px.bar(gender_distribution_dbscan, barmode='group',
                                        labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                        title='Distribuição do Gênero por Cluster - DBSCAN')
        st.plotly_chart(fig_gender_dbscan)

        st.header('Distribuição da Performance por Cluster - DBSCAN')
        fig_performance_dbscan = px.bar(performance_distribution_dbscan, barmode='group',
                                            labels={'value': 'Número de Instâncias', 'Cluster': 'Cluster'},
                                            title='Distribuição da Performance por Cluster - DBSCAN')
        st.plotly_chart(fig_performance_dbscan)


    st.subheader('13. Salvando os modelos e os dados para posterior classificação')

    with st.expander('Código e visualização dos dados', expanded=False):

        # Gerar o DataFrame final para classificação
        def generate_classification_dataframe(data, data_clustering, data_scaled, performance_col):
            # Criar um novo DataFrame com as colunas relevantes
            data_final = data_clustering.copy()

            # Adicionar as colunas de performance e outras que forem pertinentes
            data_final[performance_col] = data[performance_col]
            
            # Adicionar os rótulos gerados pelos modelos de clusterização
            data_final['cluster_kmeans'] = data['cluster_kmeans']
            data_final['cluster_agglomerative'] = data['cluster_agglomerative']
            data_final['cluster_dbscan'] = data['cluster_dbscan']

            # Incluir outras colunas do DataFrame original, se necessário
            # Exemplo: Se quiser incluir colunas categóricas que não foram usadas na clusterização
            # relevant_columns = ['other_column_1', 'other_column_2']  # Substitua pelos nomes reais das colunas
            # data_final[relevant_columns] = data[relevant_columns]

            # Salvar o DataFrame final
            data_final.to_csv('exports/data_for_classification.csv', index=False)
            
        with st.popover('Código'):
            st.code('''
                # Salvar os modelos e os dados
                config = {
                    'n_clusters': n_clusters_agglomerative,
                    'linkage': linkage_method,
                    'metric': metric_method
                }
                joblib.dump(kmeans, 'exports/kmeans_model.pkl')
                joblib.dump(config, 'exports/config_agglomerative.pkl')
                
                data_pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(n_components_pca)])
                data_pca_df.to_csv('exports/data_pca.csv', index=False)

                data_scaled_df = pd.DataFrame(data_scaled, columns=data_clustering.columns)
                data_scaled_df.to_csv('dados_normalizados.csv', index=False)

                data.to_csv('exports/data_clustered.csv', index=False)
                
                st.write(f'Foram salvos os modelos `kmeans_model.pkl` e `config_agglomerative.pkl` bem como os dados `data_pca.csv`,`dados_normalizados.csv` e `data_clustered.csv` para posterior classificação.')

                st.write(data.head(30))
                st.write('Quantidade:', len(data))
            ''')

        # Salvando os modelos e os dados
        config = {
            'n_clusters': n_clusters_agglomerative,
            'linkage': linkage_method,
            'metric': metric_method,
            'eps': eps,
            'min_samples': min_samples
        }
        joblib.dump(kmeans, 'exports/kmeans_model.pkl')
        joblib.dump(config, 'exports/config_agglomerative_dbscan.pkl')
        joblib.dump(dbscan_labels, 'exports/dbscan_model.pkl')
        
        generate_classification_dataframe(data,data_clustering,data_scaled, 'performance')

        data_pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(n_components_pca)])
        data_pca_df.to_csv('exports/data_pca.csv', index=False)

        
        data_scaled_df = pd.DataFrame(data_scaled, columns=data_clustering.columns)
        data_scaled_df['performance'] = data['performance']
        data_scaled_df.to_csv('exports/dados_normalizados.csv', index=False)

        data.to_csv('exports/data_clustered.csv', index=False)
        
        st.write(f'Modelos e dados salvos com sucesso: `config_agglomerative_dbscan.pkl`, `kmeans_model.pkl`, `dbscan_model.pkl`, `data_for_classification.csv`, `data_pca.csv`, `dados_normalizados.csv`, e `data_clustered.csv`.')

        st.write(data_scaled_df.head(30))
        st.write('Quantidade:', len(data))


    st.subheader('14. Conclusão')

    with st.expander('Código e visualização dos dados', expanded=False):

        with st.popover('Código'):
            st.code('''
                # Comparação dos Silhouette Scores
                st.subheader('Comparação dos Resultados dos Clusters')
                st.write(f"Silhouette Score - K-means: `{kmeans_silhouette}`")
                st.write(f"Silhouette Score - Agglomerative Clustering: `{ag.silhouette_score}`")    }`")

                # Exibir gráficos comparativos
                st.write("**Comparação dos Gráficos de Dispersão**")
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(scatter_fig)  # K-means
                    st.plotly_chart(fig_kmenas_distribuicao)  # K-means
                    st.plotly_chart(fig_performance)  # K-means
                with col2:
                    st.plotly_chart(scatter_fig_agglomerative)  # Agglomerative Clustering
                    st.plotly_chart(fig_agglomerative_distribution) # Agglomerative Clustering
                    st.plotly_chart(fig_performance_agglomerative)  # Agglomerative Clustering
            ''')

        # Comparação dos Silhouette Scores
        st.subheader('Comparação dos Resultados dos Clusters')
        st.write(f"Silhouette Score - K-means: `{kmeans_silhouette}`")
        st.write(f"Silhouette Score - Agglomerative Clustering: `{ag.silhouette_score}`")

        # Exibir gráficos comparativos
        st.write("**Comparação dos Gráficos de Dispersão e Distribuição**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(scatter_fig)  # K-means
            st.plotly_chart(fig_kmenas_distribuicao)  # K-means
            st.plotly_chart(fig_performance)  # K-means
        with col2:
            st.plotly_chart(scatter_fig_agglomerative)  # Agglomerative Clustering
            st.plotly_chart(fig_agglomerative_distribution) # Agglomerative Clustering
            st.plotly_chart(fig_performance_agglomerative)  # Agglomerative Clustering
        with col3:
            st.plotly_chart(fig_dbscan)
            st.plotly_chart(fig_dbscan_distribution)
            st.plotly_chart(fig_performance_dbscan)

        def let_it_rain(seconds=5):
            for _ in range(seconds):
                st.snow()
                time.sleep(1)


        if st.button("MUITO OBRIGADO!\nPASSO A PALAVRA AOS PROFESSORES!"):
            let_it_rain()


    
    # Espaçamento para empurrar o conteúdo para o rodapé
    st.sidebar.markdown(
        f"""
        <div style="height: calc(100vh - 930px);"></div>
        """,
        unsafe_allow_html=True
    )

    # Inserir a imagem no rodapé da sidebar
    st.sidebar.markdown(
        f"""
        <div style="text-align: left;">
            <img src="{gravatar_url}" alt="Gravatar" style="width: 50px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.write("## Marcelo Corni Alves")
    st.sidebar.write("Agosto/2024")
    st.sidebar.write("Disciplina: Mineração de Dados")

# Executar a aplicação Streamlit
if __name__ == '__main__':
    main()


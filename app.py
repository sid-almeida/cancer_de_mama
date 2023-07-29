# importei as bibliotecas necessárias
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import requests

# carreguei os dados
data = pd.read_csv('https://raw.githubusercontent.com/sid-almeida/cancer_de_mama/main/breast-cancer-model.csv')

# treinei o modelo
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(max_iter=1000)
model = logmodel.fit(X, y)

# criei uma função para aplicar a logaritimização dos dados inseridos
from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

def log_transform(X):
    return log_transformer.transform(X)

## Criei o app

with st.sidebar:
    st.image("https://github.com/sid-almeida/cancer_de_mama/blob/main/Brainize%20Tech%20(1).png?raw=true", width=250)
    st.title("Previsão de Câncer de Mama")
    choice = option_menu(
        None, ["Sobre", "Auto Análise", "Previsão de Diagnóstico", "Previsão de Conjunto de Dados",],
    icons=["file-person-fill","hand-index-thumb-fill", "virus", "table"])

    st.info('**Nota:** Esta aplicação foi desenvolvida com base em dados específicos e destina-se apenas a esse propósito. Por favor, evite utilizá-la para outros fins.')


if choice == "Sobre":
    st.write("""
        # Câncer de mama
        Utilizando as bibliotecas Streamlit, SCikit-Learn e Pandas. Neste aplicativo, criamos uma plataforma interativa que serve como uma ferramenta para prever o diagnóstico do tipo de câncer de mama de paciêntes baseada em dados da **OMS** coletados no estado de **Winsconsin**.
        """)
    st.write('---')
    st.write('**Sobre o App:**')
    st.write("""
        Ao utilizar o Streamlit para desenvolver o nosso aplicativo para previsão de diagnóstico de câncer de mama, 
        conseguimos criar uma interface web simples e intuitiva, proporcionando uma experiência de uso fluida e acessível. Além disso, 
        incorporamos recursos de criação de painéis interativos, permitindo que os usuários visualizem os dados de forma dinâmica.
        """)
    st.info(
        '**Nota:** Esta aplicação foi desenvolvida com base em dados específicos e destina-se apenas a esse propósito. Por favor, evite utilizá-la para outros fins.')
    st.write('---')

if choice == "Auto Análise":
    st.title("Auto Análise de Câncer de Mama")
    st.warning('**Nota:** Este aplicativo não é um substituto para a avaliação médica e um diagnóstico adequado só pode ser feito por um profissional de saúde qualificado. Se você acha que pode ter câncer de mama baseado nesta auto avaliação, consulte um médico imediatamente.')
    st.write('**Sintomas:**')
    st.info('Selecione os sitomas que você está sentindo para gerar uma probabilidade de câncer de mama.')
    st.write('---')
    x = 0

    nodulo = st.selectbox('Você sente a presença de um nódulo na mama?', ('Não', 'Sim'))
    if nodulo == 'Sim':
        x += 0.8

    dor = st.selectbox('Você sente dor na mama, mamilo ou axila?', ('Não', 'Sim'))
    if dor == 'Sim':
        x += 0.6

    tam_mama = st.selectbox('Você percebeu alguma alteração no tamanho ou formato da mama?', ('Não', 'Sim'))
    if tam_mama == 'Sim':
        x += 0.6

    form_mam = st.selectbox('Você percebeu alguma alteração no formato dos mamilos?', ('Não', 'Sim'))
    if form_mam == 'Sim':
        x += 0.7

    secrec = st.selectbox('Você percebeu alguma secreção espontânea no mamilo?', ('Não', 'Sim'))
    if secrec == 'Sim':
        x += 0.5

    pele_mama = st.selectbox(
        'Você percebeu alguma alteração no aspecto da pele da mama? (Similar a casca de uma laranja)', ('Não', 'Sim'))
    if pele_mama == 'Sim':
        x += 0.7

    irritacao_pele = st.selectbox('Você percebeu alguma irritação ou vermelhidão na pele da mama?', ('Não', 'Sim'))
    if irritacao_pele == 'Sim':
        x += 0.6

    if st.button('Gerar Probabilidade'):
        y = 4.5
        result = x / y
        st.write('---')
        st.write('**Resultado:**')
        if result < 0.5:
            st.success(f'**Probabilidade de Câncer de Mama:** {result * 100:.2f} %')
            st.warning(
                '**Nota:** Este aplicativo não é um substituto para a avaliação médica e um diagnóstico adequado só pode ser feito por um profissional de saúde qualificado. Se você acha que pode ter câncer de mama baseado nesta auto avaliação, consulte um médico imediatamente.')
        elif 0.5 <= result < 0.7:
            st.warning(
                f'**Probabilidade de Câncer de Mama:** {result * 100:.2f} %\n Procure um médico para uma avaliação mais detalhada.')
        else:
            st.error(
                f'**Probabilidade de Câncer de Mama:** {result * 100:.2f} %\n Recomendamos que você procure um médico com urgência para uma avaliação detalhada.')
            




if choice == "Previsão de Diagnóstico":
    st.title("Previsão de Diagnóstico de Câncer de Mama")
    st.write('**Diagnóstico:**')
    st.info('Insira os dados do paciente para prever o diagnóstico de câncer de mama nos campos.')
    st.write('---')

    st.write('**Dados do Paciente:**')

    radius_mean = st.number_input('Raio (Média)', min_value=0.0, max_value=100.0, value=0.0)

    texture_mean = st.number_input('Textura (Média)', min_value=0.0, max_value=100.0, value=0.0)

    perimeter_mean = st.number_input('Perímetro (Média)', min_value=0.0, max_value=100.0, value=0.0)

    smoothness_mean = st.number_input('Suavidade (Média)', min_value=0.0, max_value=100.0, value=0.0)
    log_smoothness_mean = log_transform(smoothness_mean)

    symmetry_mean = st.number_input('Simetria (Média)', min_value=0.0, max_value=100.0, value=0.0)
    log_symmetry_mean = log_transform(symmetry_mean)

    concave_points_se = st.number_input('Pontos Côncavos (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_concave_points_se = log_transform(concave_points_se)

    radius_worst = st.number_input('Raio (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_radius_worst = log_transform(radius_worst)

    perimeter_worst = st.number_input('Perímetro (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_perimeter_worst = log_transform(perimeter_worst)

    smoothness_worst = st.number_input('Suavidade (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_smoothness_worst = log_transform(smoothness_worst)

    area_mean = st.number_input('Área (Média)', min_value=0.0, max_value=100.0, value=0.0)
    log_area_mean = log_transform(area_mean)

    compactness_mean = st.number_input('Compacidade (Média)', min_value=0.0, max_value=100.0, value=0.0)
    log_compactness_mean = log_transform(compactness_mean)

    concavity_mean = st.number_input('Concavidade (Média)', min_value=0.0, max_value=100.0, value=0.0)
    log_concavity_mean = log_transform(concavity_mean)

    concave_points_mean = st.number_input('Pontos Côncavos (Média)', min_value=0.0, max_value=100.0, value=0.0)
    log_concave_points_mean = log_transform(concave_points_mean)

    fractal_dimension_mean = st.number_input('Dimensão Fractal (Média)', min_value=0.0, max_value=100.0, value=0.0)
    log_fractal_dimension_mean = log_transform(fractal_dimension_mean)

    radius_se = st.number_input('Raio (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_radius_se = log_transform(radius_se)

    texture_se = st.number_input('Textura (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_texture_se = log_transform(texture_se)

    perimeter_se = st.number_input('Perímetro (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_perimeter_se = log_transform(perimeter_se)

    area_se = st.number_input('Área (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_area_se = log_transform(area_se)

    smoothness_se = st.number_input('Suavidade (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_smoothness_se = log_transform(smoothness_se)

    compactness_se = st.number_input('Compacidade (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_compactness_se = log_transform(compactness_se)

    concavity_se = st.number_input('Concavidade (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_concavity_se = log_transform(concavity_se)

    symmetry_se = st.number_input('Simetria (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_symmetry_se = log_transform(symmetry_se)

    fractal_dimension_se = st.number_input('Dimensão Fractal (SE)', min_value=0.0, max_value=100.0, value=0.0)
    log_fractal_dimension_se = log_transform(fractal_dimension_se)

    texture_worst = st.number_input('Textura (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_texture_worst = log_transform(texture_worst)

    area_worst = st.number_input('Área Pior', min_value=0.0, max_value=100.0, value=0.0)
    log_area_worst = log_transform(area_worst)

    compactness_worst = st.number_input('Compacidade (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_compactness_worst = log_transform(compactness_worst)

    concavity_worst = st.number_input('Concavidade (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_concavity_worst = log_transform(concavity_worst)

    concave_points_worst = st.number_input('Pontos Côncavos (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_concave_points_worst = log_transform(concave_points_worst)

    symmetry_worst = st.number_input('Simetria (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_symmetry_worst = log_transform(symmetry_worst)

    fractal_dimension_worst = st.number_input('Dimensão Fractal (Pior)', min_value=0.0, max_value=100.0, value=0.0)
    log_fractal_dimension_worst = log_transform(fractal_dimension_worst)

    # Criei um botão para prever o diagnóstico
    if st.button('Prever Diagnóstico'):
        previsao = model.predict([[radius_mean, texture_mean, perimeter_mean, log_smoothness_mean, log_symmetry_mean,
                                   log_concave_points_se, log_radius_worst, log_perimeter_worst, log_smoothness_worst,
                                   log_area_mean, log_compactness_mean, log_concavity_mean, log_concave_points_mean,
                                   log_fractal_dimension_mean, log_radius_se, log_texture_se, log_perimeter_se, log_area_se,
                                   log_smoothness_se, log_compactness_se, log_concavity_se, log_symmetry_se, log_fractal_dimension_se,
                                   log_texture_worst, log_area_worst, log_compactness_worst, log_concavity_worst,
                                   log_concave_points_worst, log_symmetry_worst, log_fractal_dimension_worst]])

        probabilidade = model.predict_proba([[radius_mean, texture_mean, perimeter_mean, log_smoothness_mean, log_symmetry_mean,
                                   log_concave_points_se, log_radius_worst, log_perimeter_worst, log_smoothness_worst,
                                   log_area_mean, log_compactness_mean, log_concavity_mean, log_concave_points_mean,
                                   log_fractal_dimension_mean, log_radius_se, log_texture_se, log_perimeter_se, log_area_se,
                                   log_smoothness_se, log_compactness_se, log_concavity_se, log_symmetry_se, log_fractal_dimension_se,
                                   log_texture_worst, log_area_worst, log_compactness_worst, log_concavity_worst,
                                   log_concave_points_worst, log_symmetry_worst, log_fractal_dimension_worst]])
        if previsao == '0':
            st.warning('**Diagnóstico:** Benigno')
            st.warning(f'**Probabilidade de ser Benigno:** {probabilidade[0][0]*100} %')
        else:
            st.error('**Diagnóstico:** Maligno')
            st.error(f'**Probabilidade de ser Maligno:** {probabilidade[0][0]*100} %')


if choice == "Previsão de Conjunto de Dados":
    st.title("Previsão de Diagnóstico de Conjunto de Dados de Câncer de Mama")
    st.write('**Diagnóstico:**')
    st.info('Faça Upload do Arquivo para Realizar as Previsões ou baixe o template para criar o seu arquivo .csv.')
    st.write('---')
    
    # criei um botão para fazer o download de um template
    if st.download_button(label='Baixar Template Para Preenchimento', data=pd.read_csv('https://raw.githubusercontent.com/sid-almeida/cancer_de_mama/main/breast_pred_template.csv').to_csv(), file_name='template.csv', mime='text/csv'):
        pass
    st.write('---')
    # criei um botão para fazer o upload do arquivo
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
    if uploaded_file is not None:
        data_pred = pd.read_csv(uploaded_file, index_col=0)
        st.write('**Dados do Paciente:**')
        st.write('---')
        st.write(data_pred)
        st.write('---')

        # criei um botão para prever o diagnóstico
        if st.button('Prever Diagnóstico'):

            # apliquei a logaritimização das colunas necessárias dados inseridos
            data_pred['log_smoothness_mean'] = log_transform(data_pred['smoothness_mean'])
            data_pred['log_symmetry_mean'] = log_transform(data_pred['symmetry_mean'])
            data_pred['log_concave_points_se'] = log_transform(data_pred['concave points_se'])
            data_pred['log_radius_worst'] = log_transform(data_pred['radius_worst'])
            data_pred['log_perimeter_worst'] = log_transform(data_pred['perimeter_worst'])
            data_pred['log_smoothness_worst'] = log_transform(data_pred['smoothness_worst'])
            data_pred['log_area_mean'] = log_transform(data_pred['area_mean'])
            data_pred['log_compactness_mean'] = log_transform(data_pred['compactness_mean'])
            data_pred['log_concavity_mean'] = log_transform(data_pred['concavity_mean'])
            data_pred['log_concave points_mean'] = log_transform(data_pred['concave points_mean'])
            data_pred['log_fractal_dimension_mean'] = log_transform(data_pred['fractal_dimension_mean'])
            data_pred['log_radius_se'] = log_transform(data_pred['radius_se'])
            data_pred['log_texture_se'] = log_transform(data_pred['texture_se'])
            data_pred['log_perimeter_se'] = log_transform(data_pred['perimeter_se'])
            data_pred['log_area_se'] = log_transform(data_pred['area_se'])
            data_pred['log_smoothness_se'] = log_transform(data_pred['smoothness_se'])
            data_pred['log_compactness_se'] = log_transform(data_pred['compactness_se'])
            data_pred['log_concavity_se'] = log_transform(data_pred['concavity_se'])
            data_pred['log_symmetry_se'] = log_transform(data_pred['symmetry_se'])
            data_pred['log_fractal_dimension_se'] = log_transform(data_pred['fractal_dimension_se'])
            data_pred['log_texture_worst'] = log_transform(data_pred['texture_worst'])
            data_pred['log_area_worst'] = log_transform(data_pred['area_worst'])
            data_pred['log_compactness_worst'] = log_transform(data_pred['compactness_worst'])
            data_pred['log_concavity_worst'] = log_transform(data_pred['concavity_worst'])
            data_pred['log_concave points_worst'] = log_transform(data_pred['concave points_worst'])
            data_pred['log_symmetry_worst'] = log_transform(data_pred['symmetry_worst'])
            data_pred['log_fractal_dimension_worst'] = log_transform(data_pred['fractal_dimension_worst'])

            # removi as colunas que não estão presentes no modelo
            data_pred = data_pred.drop(['smoothness_mean', 'symmetry_mean', 'concave points_se', 'radius_worst',
                                        'perimeter_worst', 'smoothness_worst', 'area_mean', 'compactness_mean',
                                        'concavity_mean', 'concave points_mean', 'fractal_dimension_mean', 'radius_se',
                                        'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
                                        'concavity_se', 'symmetry_se', 'fractal_dimension_se', 'texture_worst',
                                        'area_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                                        'symmetry_worst', 'fractal_dimension_worst'], axis=1)

            # apliquei o modelo treinado para prever o diagnóstico
            previsao = model.predict(data_pred)
            probabilidade = model.predict_proba(data_pred)

            # criei uma tabela com os resultados
            data_pred['Diagnóstico'] = previsao
            data_pred['Probabilidade %'] = probabilidade[:, 0]*100
            st.write('**Resultado:**')
            st.write(data_pred)
            st.write('---')

            # criei um botão para baixar a tabela com os resultados
            if st.download_button(label='Baixar Resultado', data=data_pred.to_csv(), file_name='resultado.csv', mime='text/csv'):
                pass


st.write('Made with ❤️ by [Sidnei Almeida](https://www.linkedin.com/in/saaelmeida93/)')

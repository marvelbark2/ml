import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import altair as alt

st.set_option('deprecation.showPyplotGlobalUse', False)


st.set_page_config(page_title="Presentation des données",
                   layout="wide", page_icon="📈")

df = pd.read_csv("./mock_data.csv")


st.title("Machine Learning: Gestion des équipements")

st.header("Description des données")


st.subheader("Statistiques descriptives")
st.write(df.describe())


st.subheader("Statistiques descriptives: Affichage sous forme des graphes")
features = df.columns[:-1].tolist()

cols = st.columns(len(features))
for col, var in zip(cols, features):
    with col:
        plt.figure(figsize=(7, 4))  # Adjust the size as needed
        sns.histplot(df[var], kde=True)
        plt.title(f'Distribution de {var}')
        plt.xlabel(var)
        plt.ylabel('Fréquence')

        # Show plot in Streamlit
        st.pyplot(plt)
        plt.close()  # Ensure the figure is closed after rendering

st.subheader("Matrice de corrélation")
features_db = df[features]
numerical_features = features_db.select_dtypes(include=['float64', 'int64'])
numerical_features.corr()
correlation = numerical_features.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matrice de corrélation entre les variables numériques')
st.pyplot()
plt.close()

st.subheader("Matrice de corrélation")
corr_matrix = numerical_features.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
st.pyplot()
plt.close()

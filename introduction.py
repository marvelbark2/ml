import streamlit as st


st.set_page_config(page_title="Intro")


st.title("Machine Learning: Gestion des équipements")

st.header("Classification de l'équipement")

st.write("""
Nous utilisons la classification pour évaluer la qualité et l'efficacité des équipements en définissant des classes basées sur leur performance et leur fiabilité.
""")

# Classification details
st.header("Classes d'Équipement")
st.subheader("Classe A : Équipement optimal")
st.write("""
- Définition: Équipements qui fonctionnent de manière optimale, avec une performance élevée et peu de risques de panne.
""")

st.subheader("Classe B : Équipement acceptable")
st.write("""
- Définition: Équipements pouvant fonctionner correctement mais pouvant nécessiter une attention périodique pour maintenir leur performance.
""")

st.subheader("Classe C : Équipement à risque")
st.write("""
- Définition: Équipements présentant des signes évidents de défaillance imminente ou de performances inférieures à la normale, nécessitant une attention immédiate.
""")

# Algorithm to be used
st.header("Algorithme de Classification")
st.write("""
- Algorithme utilisé: **Algorithme d'apprentissage supervisé - Arbre de décision**
""")

# Equipment features
st.header("Caractéristiques de l'Équipement")
st.write("""
- Type d'équipement
- Age de l'équipement
- Durée de vie prévue de l'équipement
- État de l'équipement: bon état, besoin de réparation, besoin de maintenance
""")

# Temporal variables
st.header("Variables Temporelles")
st.write("""
- Date et heure de la dernière maintenance
- Fréquence des interventions de maintenance
- Nombre de pannes précédentes dans une période de temps donnée
""")

# Operational variables
st.header("Variables Opérationnelles")
st.write("""
- Temps de fonctionnement total de l'équipement
""")

# Performance variables
st.header("Variables de Performances")
st.write("""
- Temps moyen de réparation
- Temps d'arrêt de l'équipement en raison de pannes
""")

# Data split
st.header("Préparation des Données")
st.write("""
Nous divisons notre jeu de données en échantillons d'entraînement et de test en utilisant l'axe temporel, avec les données les plus anciennes pour l'entraînement et les plus récentes pour le test.
""")

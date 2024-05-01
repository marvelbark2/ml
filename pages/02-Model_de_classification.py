import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import os

from sklearn.model_selection import cross_val_score

st.set_page_config(page_title="Model de classification",
                   layout="wide", page_icon="🧠")

df = pd.read_csv("./mock_data.csv")

numerical_features = df.select_dtypes(include=['float64', 'int64'])

x = numerical_features
y = df["class_equipement"]


st.title("Machine Learning: Gestion des équipements")
st.header("Model de classification")

st.subheader("Arbre de décision")

model_tree = DecisionTreeClassifier()
model_tree.fit(x, y)


predictions = model_tree.predict(x)
st.metric(label="Accuracy", value=model_tree.score(x, y))
st.metric(label="Validation croisé", value=cross_val_score(
    model_tree, x, y, cv=5).mean())

df = pd.DataFrame(confusion_matrix(y, predictions),
                  columns=model_tree.classes_, index=model_tree.classes_)
st.write("Matrice de confusion")
st.scatter_chart(df)
st.table(df)

st.write("Visualisation de l'arbre de décision")

features = x.columns
fig, ax = plt.subplots(figsize=(20, 10))
decision_tree = plot_tree(
    model_tree,
    feature_names=features,
    class_names=model_tree.classes_,
    filled=True,
    rounded=True,
    node_ids=False,
)


fig.savefig("tree.svg", dpi=600)
st.image("tree.svg")
os.remove("tree.svg")

st.subheader("Arbre de décision: apprendre 0.8 et tester 0.2")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

st.metric(label="Accuracy", value=accuracy_score(y_test, predictions))
st.metric(label="Validation croisé",
          value=cross_val_score(model, x, y, cv=5).mean())
df = pd.DataFrame(confusion_matrix(y_test, predictions),
                  columns=model.classes_, index=model.classes_)

st.write("Matrice de confusion")
st.scatter_chart(df)
st.table(df)


st.write("Visualisation de l'arbre de décision")

features = x.columns
fig, ax = plt.figure(), plt.gca()
decision_tree = plot_tree(
    model,
    feature_names=features,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    node_ids=False,
)

fig.savefig("tree2.svg", dpi=600)
st.image("tree2.svg")
os.remove("tree2.svg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
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


st.subheader("Régression logistique")
model = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
model.fit(x_train, y_train)


# Replace with your actual method to get feature names if needed
feature_names = x_train.columns.tolist()
class_names = ['A', 'C', 'B']  # Adjust based on your actual class names

# Prepare to plot each feature's impact
# Adjust nrows according to the number of features
fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(10, 35))

colors = ['lightblue', 'lightgreen', 'salmon']

score = model.score(x_test, y_test)

st.metric(label="Accuracy", value=score)
st.metric(label="Validation croisé",
          value=cross_val_score(model, x, y, cv=5).mean())


st.write("Visualisation de l'impact des variables sur la prédiction")
for i, feature_name in enumerate(feature_names):
    # Create a temporary array with all rows set to the mean of each feature
    x_mean = np.tile(x_train.mean(axis=0), (300, 1))

    # Vary the i-th feature across its range
    x_range = np.linspace(
        x_train[feature_name].min(), x_train[feature_name].max(), 300)
    x_mean[:, i] = x_range

    # Predict probabilities with the i-th feature varied
    probabilities = model.predict_proba(x_mean)

    # Plot
    ax = axes[i]
    # Fill color for each class based on probabilities
    for j in range(len(colors)):
        ax.fill_between(
            x_range, 0, 1, where=probabilities[:, j] > 0.33, color=colors[j], alpha=0.3)

    # Plotting the probabilities with labels for each class
    for j in range(probabilities.shape[1]):
        ax.plot(x_range, probabilities[:, j], label=f'{
                class_names[j]}', color=colors[j])

    ax.scatter(x_train[feature_name], y_train,
               color='black', alpha=0.5, label='Data Points')
    ax.set_title(f'Impact of {feature_name} on Prediction')
    ax.set_xlabel(f'{feature_name} Values')
    ax.set_ylabel('Predicted Probability')
    ax.legend()

plt.tight_layout()
st.pyplot(fig)


st.write("Matrice de confusion")
predictions = model.predict(x_test)
df = pd.DataFrame(confusion_matrix(y_test, predictions),
                  columns=model.classes_, index=model.classes_)
st.table(df)
st.write("Matrice de confusion")
st.scatter_chart(df)


st.subheader("Random Forest")

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

st.metric(label="Accuracy", value=accuracy_score(y_test, predictions))
st.metric(label="Validation croisé",
          value=cross_val_score(model, x, y, cv=5).mean())

df = pd.DataFrame(confusion_matrix(y_test, predictions),
                  columns=model.classes_, index=model.classes_)

st.write("Matrice de confusion")
st.scatter_chart(df)

st.write("Visualisation de l'impact des variables sur la prédiction")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig = plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
        align="center")
plt.xticks(range(x_train.shape[1]), np.array(
    feature_names)[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
st.pyplot(fig)

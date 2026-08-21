# YuvaIntern 5-Week Data Science Project
# Iris Species Analysis and Classification

import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Data acquisition
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.columns = ["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm","species_id"]
df["species"] = df["species_id"].map(dict(enumerate(iris.target_names)))
df.drop(columns=["species_id"], inplace=True)

# 2. Data cleaning and EDA
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Duplicates:", df.duplicated().sum())
print(df.describe())
print(df["species"].value_counts())

# 3. Hypothesis testing
versicolor = df[df["species"]=="versicolor"]["petal_length_cm"]
virginica = df[df["species"]=="virginica"]["petal_length_cm"]

t_stat, p_value = stats.ttest_ind(versicolor, virginica, equal_var=False)
print("\nWelch t-test:", t_stat, p_value)

groups = [df[df["species"]==s]["petal_length_cm"] for s in iris.target_names]
f_stat, anova_p = stats.f_oneway(*groups)
print("ANOVA:", f_stat, anova_p)

# 4. Machine learning
features = ["sepal_length_cm","sepal_width_cm","petal_length_cm","petal_width_cm"]
X, y = df[features], df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nModel Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1:", f1_score(y_test, y_pred, average="weighted"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

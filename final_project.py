# ============================================================
# YuvaIntern 5-Week Data Science Project
# Iris Species Analysis and Classification
# ============================================================

# -----------------------------
# Import Required Libraries
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ============================================================
# WEEK 1: DATA ACQUISITION, CLEANING AND EDA
# ============================================================

print("=" * 60)
print("WEEK 1: DATA ACQUISITION, CLEANING AND EDA")
print("=" * 60)

# Load Iris dataset
iris = load_iris(as_frame=True)

df = iris.frame.copy()

# Rename columns for better readability
df.columns = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
    "species_id"
]

# Convert numeric target into species names
df["species"] = df["species_id"].map(
    dict(enumerate(iris.target_names))
)

# Remove numeric species ID
df.drop(columns=["species_id"], inplace=True)


# -----------------------------
# Dataset Overview
# -----------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isna().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nSpecies Distribution:")
print(df["species"].value_counts())


# ============================================================
# WEEK 2: DATA VISUALIZATION AND STORYTELLING
# ============================================================

print("\n" + "=" * 60)
print("WEEK 2: DATA VISUALIZATION")
print("=" * 60)


# -----------------------------
# 1. Species Distribution
# -----------------------------

species_counts = df["species"].value_counts()

plt.figure(figsize=(7, 5))

species_counts.plot(kind="bar")

plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig("species_distribution.png", dpi=300)
plt.show()


# -----------------------------
# 2. Feature Distributions
# -----------------------------

features = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm"
]

df[features].hist(figsize=(10, 8), bins=15)

plt.suptitle("Distribution of Iris Features")

plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=300)
plt.show()


# -----------------------------
# 3. Correlation Matrix
# -----------------------------

correlation_matrix = df[features].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))

plt.imshow(correlation_matrix, interpolation="nearest")
plt.colorbar()

plt.xticks(
    range(len(features)),
    features,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(features)),
    features
)

plt.title("Feature Correlation Matrix")

# Display correlation values
for i in range(len(features)):
    for j in range(len(features)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()


# -----------------------------
# 4. Petal Length vs Petal Width
# -----------------------------

plt.figure(figsize=(8, 6))

for species in iris.target_names:

    subset = df[df["species"] == species]

    plt.scatter(
        subset["petal_length_cm"],
        subset["petal_width_cm"],
        label=species
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal Length vs Petal Width by Species")

plt.legend()
plt.tight_layout()

plt.savefig(
    "petal_length_vs_width.png",
    dpi=300
)

plt.show()


# ============================================================
# WEEK 3: STATISTICAL ANALYSIS AND HYPOTHESIS TESTING
# ============================================================

print("\n" + "=" * 60)
print("WEEK 3: STATISTICAL ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# Welch's Independent Samples T-Test
# ------------------------------------------------------------
#
# Objective:
# Test whether the average petal length differs significantly
# between Iris versicolor and Iris virginica.
#
# H0:
# Mean petal lengths are equal.
#
# H1:
# Mean petal lengths are significantly different.
#
# Welch's t-test is used because it does not require equal
# population variances.
# ------------------------------------------------------------

versicolor = df[
    df["species"] == "versicolor"
]["petal_length_cm"]

virginica = df[
    df["species"] == "virginica"
]["petal_length_cm"]

t_stat, p_value = stats.ttest_ind(
    versicolor,
    virginica,
    equal_var=False
)

print("\nWelch's T-Test")
print("T-statistic:", t_stat)
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print(
        "Result: Reject H0 - "
        "There is a statistically significant difference "
        "in petal length between versicolor and virginica."
    )
else:
    print(
        "Result: Fail to reject H0 - "
        "No statistically significant difference was detected."
    )


# ------------------------------------------------------------
# One-Way ANOVA
# ------------------------------------------------------------
#
# Objective:
# Determine whether mean petal length differs among
# Setosa, Versicolor and Virginica.
#
# H0:
# All species have the same mean petal length.
#
# H1:
# At least one species has a different mean.
# ------------------------------------------------------------

groups = [
    df[df["species"] == species]["petal_length_cm"]
    for species in iris.target_names
]

f_stat, anova_p = stats.f_oneway(*groups)

print("\nOne-Way ANOVA")
print("F-statistic:", f_stat)
print("P-value:", anova_p)

if anova_p < alpha:
    print(
        "Result: Reject H0 - "
        "Petal length differs significantly across "
        "the three Iris species."
    )
else:
    print(
        "Result: Fail to reject H0."
    )


# ============================================================
# WEEK 4: MACHINE LEARNING MODEL DEVELOPMENT
# ============================================================

print("\n" + "=" * 60)
print("WEEK 4: MACHINE LEARNING MODEL")
print("=" * 60)


# -----------------------------
# Prepare Features and Target
# -----------------------------

X = df[features]
y = df["species"]


# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# -----------------------------
# Machine Learning Pipeline
# -----------------------------
#
# Step 1: StandardScaler
# Standardizes all numerical features.
#
# Step 2: LogisticRegression
# Performs multi-class Iris species classification.
# -----------------------------

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


# Train model
model.fit(X_train, y_train)


# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\nModel Evaluation")

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_test_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted"
)

print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=iris.target_names
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

error_analysis = X_test.copy()

error_analysis["Actual"] = y_test
error_analysis["Predicted"] = y_test_pred

misclassified = error_analysis[
    error_analysis["Actual"]
    != error_analysis["Predicted"]
]

print("\nNumber of Misclassified Samples:")
print(len(misclassified))

print("\nMisclassified Observations:")
print(misclassified)

if len(misclassified) > 0:

    print(
        "\nInterpretation:"
        "\nMost classification errors are expected between "
        "Versicolor and Virginica because their feature "
        "measurements overlap more than Setosa."
    )

else:

    print(
        "\nNo samples were misclassified in this test split."
    )


# ============================================================
# OVERFITTING / UNDERFITTING ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("OVERFITTING / UNDERFITTING ANALYSIS")
print("=" * 60)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

accuracy_gap = train_accuracy - test_accuracy

print("Accuracy Gap:", accuracy_gap)

if accuracy_gap > 0.10:

    print(
        "Possible Overfitting: "
        "Training accuracy is considerably higher "
        "than testing accuracy."
    )

elif train_accuracy < 0.80 and test_accuracy < 0.80:

    print(
        "Possible Underfitting: "
        "Both training and testing performance are relatively low."
    )

else:

    print(
        "The model shows a reasonable balance between "
        "training and testing performance."
    )


# ============================================================
# MODEL PERFORMANCE VISUALIZATION
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

scores = [
    test_accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(8, 5))

plt.bar(metrics, scores)

plt.ylim(0, 1)

plt.title(
    "Logistic Regression Model Performance"
)

plt.ylabel("Score")

for index, score in enumerate(scores):

    plt.text(
        index,
        score + 0.01,
        f"{score:.2f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "model_performance.png",
    dpi=300
)

plt.show()


# ============================================================
# WEEK 5: BUSINESS / PRACTICAL INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("WEEK 5: PRACTICAL INSIGHTS")
print("=" * 60)

print(
    """
Key Insights:

1. Petal measurements provide strong information for
   distinguishing Iris species.

2. Iris Setosa is generally easier to distinguish because
   its petal measurements are clearly separated from the
   other species.

3. Versicolor and Virginica show greater overlap, which can
   lead to occasional classification errors.

4. Statistical hypothesis testing confirms that petal length
   differs significantly among Iris species.

5. Logistic Regression provides a simple and interpretable
   baseline model for automated Iris species classification.

Practical Applications:

- Automated flower species identification.
- Botanical research and plant classification.
- Educational demonstrations of machine learning workflows.
- Decision-support systems based on measurable plant features.

Future Scope:

- Compare Logistic Regression with Decision Trees,
  Random Forest, K-Nearest Neighbors and Support Vector Machines.

- Perform cross-validation for more robust model evaluation.

- Conduct hyperparameter tuning.

- Add additional botanical datasets.

- Deploy the trained model through a web application or API.
"""
)


# ============================================================
# FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL PROJECT SUMMARY")
print("=" * 60)

print("Dataset Size:", df.shape)
print("Number of Features:", len(features))
print("Number of Species:", df["species"].nunique())

print(
    "Final Logistic Regression Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "\nProject successfully completed:"
    "\nData Acquisition -> Cleaning -> EDA -> Visualization ->"
    "\nStatistical Analysis -> Machine Learning -> Evaluation ->"
    "\nError Analysis -> Practical Insights"
)

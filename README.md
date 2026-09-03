# 🌸 Iris Species Analysis and Classification

A complete 5-week Data Science with Python project developed as part of the **YuvaIntern Data Science Internship**.

This project demonstrates an end-to-end data science workflow using the classic Iris dataset, covering data acquisition, data cleaning, exploratory data analysis, visualization, statistical hypothesis testing, machine learning, model evaluation, error analysis, and practical recommendations.

---

## 📌 Project Overview

The objective of this project is to analyze Iris flower measurements and build a machine learning model capable of classifying flowers into one of three species:

- 🌱 Iris Setosa
- 🌸 Iris Versicolor
- 🌺 Iris Virginica

The project combines statistical analysis and machine learning to understand which measurements are most useful for distinguishing different Iris species.

---

## 🎯 Project Objectives

The major objectives of this project are:

1. Acquire and prepare the Iris dataset.
2. Perform data cleaning and exploratory data analysis.
3. Analyze feature distributions and relationships.
4. Visualize important patterns in the dataset.
5. Perform statistical hypothesis testing.
6. Build a machine learning classification pipeline.
7. Evaluate model performance using multiple metrics.
8. Analyze classification errors.
9. Investigate possible overfitting and underfitting.
10. Derive practical insights and future improvements.

---

# 🗂️ Project Structure

```text
YuvaIntern-Data-Science-with-Python/
│
├── README.md
├── final_project.py
├── requirements.txt
│
├── species_distribution.png
├── feature_distributions.png
├── correlation_matrix.png
├── petal_length_vs_width.png
├── confusion_matrix.png
└── model_performance.png
```

> The visualization files are generated automatically when `final_project.py` is executed.

---

# 📊 Dataset

The project uses the **Iris dataset** provided by Scikit-learn.

The dataset contains:

| Property | Value |
|---|---:|
| Total observations | 150 |
| Features | 4 |
| Species | 3 |
| Samples per species | 50 |
| Missing values | 0 |

### Features

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

### Target

The target variable is the Iris species:

```text
setosa
versicolor
virginica
```

---

# 🔬 Methodology

The project follows a complete data science workflow:

```text
Data Acquisition
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Data Visualization
       ↓
Statistical Hypothesis Testing
       ↓
Train-Test Split
       ↓
Feature Scaling
       ↓
Logistic Regression
       ↓
Model Evaluation
       ↓
Error Analysis
       ↓
Practical Insights
```

---

# 📅 Week 1 — Data Acquisition, Cleaning and EDA

## Data Acquisition

The Iris dataset is loaded using Scikit-learn:

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame.copy()
```

The dataset is then converted into a Pandas DataFrame and the columns are renamed for readability.

## Data Cleaning

The following checks were performed:

- Missing value detection
- Duplicate row detection
- Dataset structure inspection
- Descriptive statistics
- Species distribution analysis

Example:

```python
print(df.isna().sum())
print(df.duplicated().sum())
print(df.describe())
```

### EDA Findings

The dataset contains measurements for three Iris species.

The analysis shows that petal measurements provide strong information for distinguishing the species, while some measurements show greater overlap between Versicolor and Virginica.

---

# 📈 Week 2 — Data Visualization and Storytelling

Multiple visualizations were created to understand the dataset.

## 1. Species Distribution

A bar chart shows the number of observations belonging to each species.

```python
species_counts = df["species"].value_counts()

species_counts.plot(kind="bar")

plt.title("Distribution of Iris Species")
plt.xlabel("Species")
plt.ylabel("Number of Samples")
```

## 2. Feature Distributions

Histograms are used to understand the distribution of:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

```python
df[features].hist(figsize=(10, 8), bins=15)
```

## 3. Correlation Analysis

A correlation matrix is used to identify relationships between numerical features.

```python
correlation_matrix = df[features].corr()
```

This helps identify which measurements are strongly related to each other.

## 4. Petal Length vs Petal Width

A scatter plot compares petal length and petal width across the three species.

This visualization helps demonstrate why petal measurements are useful for classification.

---

# 🧪 Week 3 — Statistical Analysis and Hypothesis Testing

Statistical testing was performed to determine whether differences observed in the dataset are statistically significant.

## Welch's Independent Samples T-Test

The first test compares petal length between:

- Iris Versicolor
- Iris Virginica

### Null Hypothesis (H₀)

The mean petal length of Versicolor and Virginica is equal.

### Alternative Hypothesis (H₁)

The mean petal length of Versicolor and Virginica is different.

### Method

Welch's t-test was selected because it does not require the two groups to have equal population variances.

```python
t_stat, p_value = stats.ttest_ind(
    versicolor,
    virginica,
    equal_var=False
)
```

A significance level of:

```text
α = 0.05
```

was used.

If:

```text
p-value < 0.05
```

the null hypothesis is rejected.

### Interpretation

The test indicates a statistically significant difference in petal length between Versicolor and Virginica.

This supports the visual observation that petal measurements can help distinguish Iris species.

---

## One-Way ANOVA

A one-way ANOVA was performed to compare petal length across all three species.

### Null Hypothesis (H₀)

All three species have the same mean petal length.

### Alternative Hypothesis (H₁)

At least one species has a different mean petal length.

```python
groups = [
    df[df["species"] == species]["petal_length_cm"]
    for species in iris.target_names
]

f_stat, anova_p = stats.f_oneway(*groups)
```

### Interpretation

The ANOVA result indicates statistically significant differences in petal length across the Iris species.

This provides statistical evidence that species classification can be supported by measurable differences in flower characteristics.

---

# 🤖 Week 4 — Machine Learning Model

## Model Used

A **Logistic Regression** classifier was developed as the baseline machine learning model.

Before training, the data was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Dataset Split

```text
80% → Training
20% → Testing
```

The `stratify=y` parameter maintains a balanced representation of all three species in both sets.

---

# ⚙️ Machine Learning Pipeline

A Scikit-learn Pipeline was used to combine preprocessing and model training.

```python
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
```

### Pipeline Steps

### Step 1 — Standardization

`StandardScaler` standardizes numerical features so that they are placed on a comparable scale.

### Step 2 — Logistic Regression

Logistic Regression performs multi-class classification and predicts the species of an Iris flower based on its measurements.

### Step 3 — Prediction

```python
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
```

---

# 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

```python
accuracy = accuracy_score(y_test, y_test_pred)

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
```

## Performance Results

> **Important:** The values below should be updated with the exact output produced by the latest version of `final_project.py`.

| Metric | Result |
|---|---:|
| Accuracy | Run latest code |
| Precision | Run latest code |
| Recall | Run latest code |
| F1 Score | Run latest code |

The latest execution output should be used as the final reported result.

---

# 🔍 Error Analysis

Model predictions are compared with the actual species labels to identify incorrectly classified observations.

```python
error_analysis = X_test.copy()

error_analysis["Actual"] = y_test
error_analysis["Predicted"] = y_test_pred

misclassified = error_analysis[
    error_analysis["Actual"] != error_analysis["Predicted"]
]
```

This allows individual classification errors to be inspected rather than relying only on an overall accuracy score.

### Practical Interpretation

Classification errors are expected to occur mainly where species have overlapping physical measurements.

In particular, **Versicolor and Virginica** have more similar measurements than Setosa, making their classification comparatively more challenging.

---

# 🧠 Overfitting and Underfitting Analysis

Training and testing performance are compared to determine whether the model generalizes well.

```python
train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

accuracy_gap = train_accuracy - test_accuracy
```

### Interpretation

A large difference between training and testing accuracy can indicate possible overfitting.

If both training and testing performance are low, the model may be underfitting.

A small performance gap with strong testing performance indicates better generalization.

The latest execution of the project reports the actual training and testing accuracy values.

---

# 📉 Confusion Matrix

The confusion matrix provides a detailed view of correct and incorrect predictions for each species.

It helps answer questions such as:

- How many Setosa samples were classified correctly?
- How many Versicolor samples were confused with Virginica?
- Which species is hardest for the model to distinguish?

This provides more detailed information than accuracy alone.

---

# 💼 Practical / Business Implications

Although Iris is an educational dataset, the same workflow can be applied to real-world classification problems.

### Potential Applications

🌱 **Plant Identification**

Automated identification of plant species based on measurable characteristics.

🔬 **Botanical Research**

Statistical analysis can help researchers understand differences between species.

🤖 **Automated Classification**

Machine learning can reduce manual classification effort when measurable features are available.

📚 **Educational Applications**

The project demonstrates a complete machine learning workflow for students and beginners.

### General Business Value

The same approach can be adapted to business problems such as:

- Customer classification
- Fraud detection
- Customer churn prediction
- Product categorization
- Medical image classification
- Quality-control systems

The important principle is to combine **data analysis + statistical evidence + predictive modeling** rather than relying on a single technique.

---

# 🚀 Future Scope

The project can be improved in several ways.

## 1. Compare Multiple Algorithms

Future versions can compare:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine

## 2. Cross-Validation

K-fold cross-validation can provide a more robust estimate of model performance.

## 3. Hyperparameter Tuning

Techniques such as Grid Search or Randomized Search can be used to optimize model parameters.

## 4. Feature Selection

Feature importance and selection techniques can identify the most useful measurements.

## 5. Model Deployment

The final model could be deployed using:

- Streamlit
- Flask
- FastAPI

A user could enter flower measurements and receive a predicted species.

## 6. Larger Real-World Dataset

Future work can use larger and more complex botanical datasets to evaluate whether the approach scales to real-world problems.

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| SciPy | Statistical testing |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/udai33/YuvaIntern-Data-Science-with-Python.git
```

Move into the project directory:

```bash
cd YuvaIntern-Data-Science-with-Python
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python final_project.py
```

---

# 📁 Generated Visualizations

Running `final_project.py` generates the following visualizations:

```text
species_distribution.png
feature_distributions.png
correlation_matrix.png
petal_length_vs_width.png
confusion_matrix.png
model_performance.png
```

These visualizations can be uploaded to the repository and displayed in this README.

---

# 🏆 Key Takeaways

The project demonstrates that:

- Data cleaning is essential before analysis.
- Visualization helps identify patterns that may not be obvious from raw data.
- Statistical hypothesis testing provides quantitative evidence for observed differences.
- Petal measurements are particularly useful for distinguishing Iris species.
- Logistic Regression provides a strong and interpretable baseline classification model.
- Confusion matrices and error analysis provide deeper insights than accuracy alone.
- Comparing training and testing performance helps evaluate model generalization.
- The complete workflow can be adapted to many real-world classification problems.

---

# 👨‍💻 Author

**Udai Singh Negi**

B.Tech Computer Science & Engineering (AI)

GitHub: **udai33**

---

# 📜 Internship

This project was developed as part of the:

**YuvaIntern — Data Science with Python Internship**

The project covers a five-week data science workflow from data acquisition to machine learning evaluation and strategic recommendations.

---

⭐ If you find this project useful, feel free to explore the repository and the implementation in `final_project.py`.

# PAKISTAN-ODI-CRICKET-STATISTICS-2019-2023
Absolutely. This dataset is suitable for a **portfolio-level Data Science project**, but there is one important limitation: the uploaded file is the **Kaggle metadata file**, not the actual CSV data. The metadata confirms that the underlying dataset contains **43 ODI match records** and the main fields are `Team 1`, `Team 2`, `Format`, `Date`, `Venue`, `Result`, and `Status`. 

So I can build the project structure and complete Python code around the actual dataset schema, but I don't want to invent match records or fake model results.

## 📊 Portfolio Project

### **Pakistan ODI Cricket Performance Analysis & Match Outcome Prediction**

**Goal:** Analyze Pakistan's ODI performance from 2019–2023 and build a machine-learning pipeline to predict match outcomes based on available match information.

The project will demonstrate:

* Data loading
* Data cleaning
* Exploratory Data Analysis (EDA)
* Statistical analysis
* Feature engineering
* Data visualization
* Correlation/category analysis
* Machine Learning
* Classification
* Model comparison
* Hyperparameter tuning
* Cross-validation
* Feature importance
* Prediction
* Clustering
* Model evaluation
* Business/analytical insights
* Portfolio-quality conclusions

The dataset specifically covers Pakistan ODI matches played in Pakistan and contains information about teams, dates, venues, results and match status. 

---

# 1. Project Structure

I recommend organizing your GitHub project like this:

```text
pakistan-odi-data-science/
│
├── data/
│   └── ODI statistics in Pakistan 2019-23.csv
│
├── notebooks/
│   └── Pakistan_ODI_Data_Science_Project.ipynb
│
├── src/
│   └── prediction.py
│
├── visualizations/
│   ├── matches_by_year.png
│   ├── venue_analysis.png
│   ├── team_performance.png
│   └── model_comparison.png
│
├── README.md
├── requirements.txt
└── Pakistan_ODI_Project_Report.pdf
```

---

# 2. Complete Notebook

## Cell 1 — Import Libraries

```python
# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Clustering
from sklearn.cluster import KMeans

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print("Libraries imported successfully.")
```

---

# 3. Load Dataset

```python
df = pd.read_csv("../data/ODI statistics in Pakistan 2019-23.csv")

df.head()
```

If your CSV is in the same directory as the notebook:

```python
df = pd.read_csv("ODI statistics in Pakistan 2019-23.csv")
```

---

# 4. Understand the Dataset

```python
print("Shape:", df.shape)
```

```python
df.info()
```

```python
df.columns
```

```python
df.describe(include="all")
```

```python
df.head(10)
```

The expected variables are:

```text
Team 1
Team 2
Format
Date
Venue
Result
Status
```

These are the seven fields documented in the uploaded dataset metadata. 

---

# 5. Data Quality Analysis

## Missing Values

```python
missing_values = df.isnull().sum()

print(missing_values)
```

Percentage:

```python
missing_percentage = (
    df.isnull().sum() / len(df) * 100
).sort_values(ascending=False)

missing_percentage
```

Visualization:

```python
plt.figure(figsize=(10, 5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Heatmap")
plt.show()
```

---

# 6. Duplicate Records

```python
print("Duplicate rows:", df.duplicated().sum())
```

Remove duplicates:

```python
df = df.drop_duplicates()

print("New shape:", df.shape)
```

---

# 7. Date Processing

Convert date:

```python
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
```

Create useful features:

```python
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
```

Check:

```python
df[["Date", "Year", "Month", "Day", "DayOfWeek"]].head()
```

---

# 8. Exploratory Data Analysis

## Matches by Year

```python
year_counts = df["Year"].value_counts().sort_index()

plt.figure(figsize=(10, 5))

year_counts.plot(kind="bar")

plt.title("Pakistan ODI Matches by Year")
plt.xlabel("Year")
plt.ylabel("Number of Matches")

plt.xticks(rotation=0)
plt.show()
```

---

# 9. Venue Analysis

```python
venue_counts = df["Venue"].value_counts()

venue_counts
```

Visualization:

```python
plt.figure(figsize=(12, 6))

venue_counts.plot(kind="bar")

plt.title("ODI Matches by Venue")
plt.xlabel("Venue")
plt.ylabel("Number of Matches")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.show()
```

---

# 10. Opponent Analysis

Because the dataset has `Team 1` and `Team 2`, we can analyze Pakistan's opponents.

```python
teams = pd.concat([
    df["Team 1"],
    df["Team 2"]
])

teams.value_counts()
```

---

# 11. Result Analysis

```python
df["Result"].value_counts()
```

Visualization:

```python
plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="Result"
)

plt.title("Pakistan ODI Match Results")
plt.xlabel("Result")
plt.ylabel("Number of Matches")

plt.xticks(rotation=45)

plt.show()
```

---

# 12. Status Analysis

```python
df["Status"].value_counts()
```

```python
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Status"
)

plt.title("Match Status Distribution")

plt.xticks(rotation=45)

plt.show()
```

This is particularly important because the dataset includes completed, abandoned and canceled games. 

---

# 13. Pakistan Win/Loss Feature

Because `Result` may contain different text values, first inspect it:

```python
print(df["Result"].unique())
```

Then create a target variable based on the actual values in your CSV.

For example, if Pakistan's wins appear as `"Pakistan won"`:

```python
df["Pakistan_Win"] = (
    df["Result"].str.contains(
        "Pakistan",
        case=False,
        na=False
    )
    &
    ~df["Result"].str.contains(
        "lost|loss",
        case=False,
        na=False
    )
).astype(int)
```

**Important:** inspect the actual `Result` values before using this transformation. We should not assume the exact wording from metadata alone.

---

# 14. Feature Engineering

Create an opponent feature.

```python
def get_opponent(row):
    if str(row["Team 1"]).lower() == "pakistan":
        return row["Team 2"]
    else:
        return row["Team 1"]

df["Opponent"] = df.apply(get_opponent, axis=1)
```

Check:

```python
df[[
    "Team 1",
    "Team 2",
    "Opponent",
    "Venue",
    "Result"
]].head()
```

---

# 15. Feature: Pakistan Batting First

```python
df["Pakistan_Batting_First"] = np.where(
    df["Team 1"].str.lower() == "pakistan",
    1,
    0
)
```

This gives us a simple predictive feature.

---

# 16. Statistical Analysis

Opponent performance:

```python
opponent_summary = (
    df.groupby("Opponent")
      .agg(
          Matches=("Opponent", "count")
      )
      .sort_values(
          "Matches",
          ascending=False
      )
)

opponent_summary
```

Venue frequency:

```python
venue_summary = (
    df.groupby("Venue")
      .size()
      .sort_values(ascending=False)
)

venue_summary
```

Yearly matches:

```python
year_summary = (
    df.groupby("Year")
      .size()
)

year_summary
```

---

# 17. Machine Learning Problem

We can formulate:

### Target

```text
Pakistan_Win
```

### Features

```text
Opponent
Venue
Year
Month
DayOfWeek
Pakistan_Batting_First
```

This becomes a **binary classification problem**.

```text
0 = Pakistan did not win
1 = Pakistan won
```

---

# 18. Prepare Machine Learning Data

```python
features = [
    "Opponent",
    "Venue",
    "Year",
    "Month",
    "DayOfWeek",
    "Pakistan_Batting_First"
]

X = df[features]
y = df["Pakistan_Win"]
```

---

# 19. Train/Test Split

Because the dataset is very small, use stratification if both classes have enough samples.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

---

# 20. Preprocessing Pipeline

Categorical features:

```python
categorical_features = [
    "Opponent",
    "Venue"
]
```

Numerical features:

```python
numerical_features = [
    "Year",
    "Month",
    "DayOfWeek",
    "Pakistan_Batting_First"
]
```

Create preprocessing:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)
```

---

# 21. Logistic Regression

```python
logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression())
    ]
)

logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_test)
```

Evaluation:

```python
print(
    classification_report(
        y_test,
        y_pred_lr
    )
)
```

---

# 22. Decision Tree

```python
decision_tree = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeClassifier(
                max_depth=4,
                random_state=42
            )
        )
    ]
)

decision_tree.fit(X_train, y_train)

y_pred_dt = decision_tree.predict(X_test)
```

```python
print(
    classification_report(
        y_test,
        y_pred_dt
    )
)
```

---

# 23. Random Forest

```python
random_forest = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)
```

```python
print(
    classification_report(
        y_test,
        y_pred_rf
    )
)
```

---

# 24. Gradient Boosting

```python
gradient_boosting = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            GradientBoostingClassifier(
                random_state=42
            )
        )
    ]
)

gradient_boosting.fit(X_train, y_train)

y_pred_gb = gradient_boosting.predict(X_test)
```

---

# 25. Compare Models

```python
models = {
    "Logistic Regression": y_pred_lr,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gb
}

results = []

for name, predictions in models.items():

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    })

model_results = pd.DataFrame(results)

model_results.sort_values(
    "F1 Score",
    ascending=False
)
```

---

# 26. Model Comparison Visualization

```python
model_results.set_index(
    "Model"
)[[
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
```

---

# 27. Confusion Matrix

For the best-performing model:

```python
best_model = random_forest

y_pred = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred
)

ConfusionMatrixDisplay(
    confusion_matrix=cm
).plot()

plt.title("Confusion Matrix")
plt.show()
```

---

# 28. Cross Validation

Since only a small number of matches are available, a single train/test split can be unstable.

Use cross-validation:

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

```python
scores = cross_val_score(
    random_forest,
    X,
    y,
    cv=cv,
    scoring="f1"
)

print("Cross-validation F1 scores:")
print(scores)

print(
    "Mean F1:",
    scores.mean()
)
```

This is much better for a portfolio project because you demonstrate that you understand that **small datasets require careful evaluation**.

---

# 29. Hyperparameter Tuning

```python
param_grid = {
    "model__n_estimators": [
        100,
        200,
        300
    ],
    "model__max_depth": [
        3,
        5,
        10,
        None
    ],
    "model__min_samples_split": [
        2,
        5,
        10
    ]
}
```

```python
grid_search = GridSearchCV(
    random_forest,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
```

Best parameters:

```python
print(
    "Best Parameters:",
    grid_search.best_params_
)
```

Best score:

```python
print(
    "Best CV Score:",
    grid_search.best_score_
)
```

---

# 30. Prediction System

We can create a function that predicts a future match.

```python
def predict_match(
    opponent,
    venue,
    year,
    month,
    day_of_week,
    batting_first
):

    new_match = pd.DataFrame({
        "Opponent": [opponent],
        "Venue": [venue],
        "Year": [year],
        "Month": [month],
        "DayOfWeek": [day_of_week],
        "Pakistan_Batting_First": [batting_first]
    })

    prediction = grid_search.predict(
        new_match
    )

    probability = grid_search.predict_proba(
        new_match
    )

    if prediction[0] == 1:
        result = "Pakistan Win"
    else:
        result = "Pakistan Not Win"

    return result, probability[0]
```

Example:

```python
predict_match(
    opponent="New Zealand",
    venue="Gaddafi Stadium",
    year=2023,
    month=4,
    day_of_week=5,
    batting_first=1
)
```

---

# 31. Clustering — Unsupervised Learning

We can also demonstrate **unsupervised learning**.

For example, create venue-level statistics.

```python
venue_features = (
    df.groupby("Venue")
      .agg(
          Matches=("Venue", "count"),
          Years=("Year", "nunique")
      )
      .reset_index()
)

venue_features
```

Scale:

```python
scaler = StandardScaler()

X_cluster = scaler.fit_transform(
    venue_features[
        ["Matches", "Years"]
    ]
)
```

K-Means:

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

venue_features["Cluster"] = kmeans.fit_predict(
    X_cluster
)
```

```python
venue_features
```

Visualization:

```python
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=venue_features,
    x="Matches",
    y="Years",
    hue="Cluster",
    palette="viridis",
    s=100
)

plt.title("Venue Clustering")
plt.show()
```

This gives your project an additional **unsupervised machine-learning component**.

---

# 32. Advanced Analysis — Win Rate by Opponent

Once the target is correctly created:

```python
opponent_win_rate = (
    df.groupby("Opponent")["Pakistan_Win"]
      .agg(
          Matches="count",
          Wins="sum"
      )
)

opponent_win_rate["Win_Rate"] = (
    opponent_win_rate["Wins"]
    /
    opponent_win_rate["Matches"]
    * 100
)

opponent_win_rate.sort_values(
    "Win_Rate",
    ascending=False
)
```

---

# 33. Win Rate by Venue

```python
venue_win_rate = (
    df.groupby("Venue")["Pakistan_Win"]
      .agg(
          Matches="count",
          Wins="sum"
      )
)

venue_win_rate["Win_Rate"] = (
    venue_win_rate["Wins"]
    /
    venue_win_rate["Matches"]
    * 100
)

venue_win_rate.sort_values(
    "Win_Rate",
    ascending=False
)
```

---

# 34. Yearly Performance

```python
year_performance = (
    df.groupby("Year")["Pakistan_Win"]
      .agg(
          Matches="count",
          Wins="sum"
      )
)

year_performance["Win_Rate"] = (
    year_performance["Wins"]
    /
    year_performance["Matches"]
    * 100
)

year_performance
```

Plot:

```python
plt.figure(figsize=(10, 5))

plt.plot(
    year_performance.index,
    year_performance["Win_Rate"],
    marker="o"
)

plt.title("Pakistan ODI Win Rate by Year")
plt.xlabel("Year")
plt.ylabel("Win Rate (%)")

plt.grid(True)

plt.show()
```

---

# 35. Portfolio Dashboard Ideas

You can later turn this into a **Streamlit dashboard**.

Dashboard sections:

```text
Pakistan ODI Analytics Dashboard
│
├── Overview
│   ├── Total Matches
│   ├── Wins
│   ├── Losses
│   └── Win Rate
│
├── Performance Analysis
│   ├── Yearly Performance
│   ├── Opponent Performance
│   └── Venue Performance
│
├── Visual Analytics
│   ├── Match Distribution
│   ├── Result Distribution
│   └── Venue Analysis
│
├── Machine Learning
│   ├── Model Comparison
│   ├── Confusion Matrix
│   └── Feature Importance
│
└── Match Predictor
    ├── Select Opponent
    ├── Select Venue
    ├── Enter Date
    ├── Batting First
    └── Predict Outcome
```

---

# 36. Streamlit Application

After the notebook works, create:

`app.py`

```python
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Pakistan ODI Analytics",
    page_icon="🏏",
    layout="wide"
)

st.title("🏏 Pakistan ODI Cricket Analytics")

st.write(
    "Data Science and Machine Learning Analysis "
    "of Pakistan ODI matches"
)

df = pd.read_csv(
    "data/ODI statistics in Pakistan 2019-23.csv"
)

st.subheader("Dataset")

st.dataframe(df)

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Matches",
        len(df)
    )

with col2:
    st.metric(
        "Venues",
        df["Venue"].nunique()
    )

with col3:
    st.metric(
        "Opponents",
        len(
            set(df["Team 1"])
            .union(set(df["Team 2"]))
        )
    )

st.subheader("Results")

result_counts = df["Result"].value_counts()

st.bar_chart(result_counts)
```

Run:

```bash
streamlit run app.py
```

---

# 37. Requirements File

Create:

`requirements.txt`

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
streamlit
```

Install:

```bash
pip install -r requirements.txt
```

---

# 38. What Makes This a Good Portfolio Project?

Instead of presenting it simply as:

> "I analyzed a cricket dataset."

Present it as:

> **End-to-End Sports Analytics & Machine Learning Project**

You demonstrate:

| Data Science Skill  | Project Component              |
| ------------------- | ------------------------------ |
| Python              | Entire project                 |
| Pandas              | Data manipulation              |
| NumPy               | Numerical processing           |
| Data Cleaning       | Missing/duplicate handling     |
| EDA                 | Match/venue/opponent analysis  |
| Visualization       | Matplotlib + Seaborn           |
| Feature Engineering | Date/opponent/batting features |
| Statistics          | Win-rate analysis              |
| Classification      | Match outcome prediction       |
| Logistic Regression | ML model                       |
| Decision Tree       | ML model                       |
| Random Forest       | ML model                       |
| Gradient Boosting   | ML model                       |
| Cross Validation    | Model validation               |
| Grid Search         | Hyperparameter tuning          |
| Confusion Matrix    | Evaluation                     |
| F1/Precision/Recall | Evaluation metrics             |
| K-Means             | Clustering                     |
| Streamlit           | Deployment                     |
| GitHub              | Portfolio presentation         |

---

## ⚠️ One important issue with the dataset

There is a limitation that should actually be mentioned in your portfolio rather than hidden.

The uploaded metadata says the dataset contains **43 match-level records**, while the available variables are mainly categorical/date information such as teams, venue, result and status. 

That means this is **not a strong dataset for a production-grade match prediction model**. For example, it does not appear to contain:

* Player statistics
* Runs scored
* Wickets
* Overs
* Run rate
* Bowling economy
* Toss result
* Pitch conditions
* Weather
* Player form
* Batting averages

Therefore, I would **not claim 90–95% prediction accuracy** just to make the portfolio look impressive. With only 43 observations, that would likely be misleading.

### A stronger version would be:

**Project title:**

> 🏏 **Pakistan ODI Performance Analytics & Machine Learning Prediction System**

**Techniques:**

```text
Data Cleaning
       ↓
EDA
       ↓
Statistical Analysis
       ↓
Feature Engineering
       ↓
Data Visualization
       ↓
Classification
       ↓
Model Comparison
       ↓
Cross Validation
       ↓
Hyperparameter Tuning
       ↓
K-Means Clustering
       ↓
Match Prediction
       ↓
Streamlit Dashboard
```

The dataset's own description specifically identifies **data analytics, data visualization and AI/data-science applications**, and states that it can be used for analyzing trends and predicting outcomes. 

**If you upload the actual `ODI statistics in Pakistan 2019-23.csv` file (rather than only the metadata JSON), I can take this one step further and build the complete runnable project from the real 43 records—including the actual EDA charts, actual model scores, feature analysis, prediction results, and a polished Jupyter Notebook ready for your GitHub portfolio.**

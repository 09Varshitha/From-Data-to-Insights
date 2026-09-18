import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    r"C:\Users\Varshitha\OneDrive\Documents\Desktop\ml project\Prescription_Monitoring_Program__Opioid__Benzodiazepine__Stimulant__Naloxone__and_Gabapentin_Prescription_Count___Rate_by_Patient_Residence.csv"
)

print("First five rows:\n", df.head())
print("\nDataset Info:")
print(df.info())


# ============================================================
# DATA CLEANING
# ============================================================

df.fillna(df.mean(numeric_only=True), inplace=True)

duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)

print("Number of duplicate rows removed:", duplicates)
print("Missing Values after cleaning:", df.isnull().sum().sum())
print("Statistical Summary:\n", df.describe())


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df['Total Prescriptions'] = (
    df['Opiate Agonist Prescription Count'] +
    df['Opiate Partial Agonist Prescription Count'] +
    df['Benzodiazepine Prescription Count'] +
    df['Stimulant Prescription Count'] +
    df['Gabapentin Prescription Count']
)


# ============================================================
# OBJECTIVE 1:
# ANALYZE RELATIONSHIP BETWEEN DRUG PRESCRIPTIONS
# ============================================================

drug_cols = [
    'Opiate Agonist Prescription Count',
    'Opiate Partial Agonist Prescription Count',
    'Benzodiazepine Prescription Count',
    'Stimulant Prescription Count',
    'Gabapentin Prescription Count',
    'Naloxone Prescription Count'
]

plt.figure(figsize=(14, 10))

sns.heatmap(
    df[drug_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title(
    "Correlation Analysis Between Drug Prescription Types",
    fontsize=16
)

plt.xticks(rotation=10, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 2:
# ANALYZE PRESCRIPTION TRENDS OVER TIME
# ============================================================

df["Year"] = df["Date"].str.extract(
    r'(\d{4})'
).astype(int)

df["Quarter"] = df["Date"].str.extract(
    r'Quarter (\d)'
).astype(int)

df["Time_Index"] = (
    df["Year"] +
    (df["Quarter"] - 1) / 4
)

trend = df.groupby(
    "Time_Index"
)["Total Prescriptions"].sum().reset_index()


X = trend[["Time_Index"]]
y = trend["Total Prescriptions"]


poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()

poly_model.fit(
    X_poly,
    y
)

y_poly_pred = poly_model.predict(
    X_poly
)


plt.figure(figsize=(9, 5))

plt.scatter(
    X,
    y,
    label="Actual Prescriptions"
)

plt.plot(
    X,
    y_poly_pred,
    color="green",
    linewidth=2,
    label="Polynomial Trend (Degree 2)"
)

plt.xlabel("Time (Year + Quarter)")
plt.ylabel("Total Prescriptions")

plt.title(
    "Prescription Trend Using Polynomial Regression"
)

plt.legend()
plt.tight_layout()
plt.show()


print(
    "Polynomial Trend R²:",
    r2_score(y, y_poly_pred)
)

print(
    "Polynomial Trend RMSE:",
    np.sqrt(
        mean_squared_error(
            y,
            y_poly_pred
        )
    )
)


# ============================================================
# OBJECTIVE 3:
# PREDICT NALOXONE PRESCRIPTION COUNT USING REGRESSION
# ============================================================

features = [
    'Opiate Agonist Prescription Count',
    'Opiate Partial Agonist Prescription Count',
    'Benzodiazepine Prescription Count',
    'Stimulant Prescription Count',
    'Gabapentin Prescription Count',
    'Total Prescriptions'
]

X = df[features]

# Target is the actual numerical Naloxone Prescription Count
y = df['Naloxone Prescription Count']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)


print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        linear_pred
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            linear_pred
        )
    )
)

print(
    "R²:",
    r2_score(
        y_test,
        linear_pred
    )
)


# ============================================================
# POLYNOMIAL REGRESSION
# ============================================================

poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

X_train_poly = poly.fit_transform(
    X_train
)

X_test_poly = poly.transform(
    X_test
)

poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train
)

poly_pred = poly_model.predict(
    X_test_poly
)


print("\n" + "=" * 60)
print("POLYNOMIAL REGRESSION")
print("=" * 60)

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        poly_pred
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            poly_pred
        )
    )
)

print(
    "R²:",
    r2_score(
        y_test,
        poly_pred
    )
)


# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)


print("\n" + "=" * 60)
print("RANDOM FOREST REGRESSION")
print("=" * 60)

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        rf_pred
    )
)

print(
    "RMSE:",
    np.sqrt(
        mean_squared_error(
            y_test,
            rf_pred
        )
    )
)

print(
    "R²:",
    r2_score(
        y_test,
        rf_pred
    )
)


# ============================================================
# ACTUAL VS PREDICTED - RANDOM FOREST
# ============================================================

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    rf_pred,
    alpha=0.6
)

plt.xlabel(
    "Actual Naloxone Prescription Count"
)

plt.ylabel(
    "Predicted Naloxone Prescription Count"
)

plt.title(
    "Random Forest Regression: Actual vs Predicted"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.tight_layout()
plt.show()


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)

print(importance)


plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 4:
# K-MEANS CLUSTERING
# ============================================================

from sklearn.cluster import KMeans

drug_features = [
    'Opiate Agonist Prescription Count',
    'Benzodiazepine Prescription Count',
    'Stimulant Prescription Count',
    'Gabapentin Prescription Count',
    'Naloxone Prescription Count'
]

X_drugs = df[drug_features]

X_scaled = StandardScaler().fit_transform(
    X_drugs
)

wcss = []

for k in range(1, 11):

    km = KMeans(
        n_clusters=k,
        random_state=42
    )

    km.fit(X_scaled)

    wcss.append(
        km.inertia_
    )


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "WCSS (Inertia)"
)

plt.title(
    "Elbow Method for Optimal Number of Clusters"
)

plt.tight_layout()
plt.show()


kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

df['Drug_Cluster'] = kmeans.fit_predict(
    X_scaled
)


from sklearn.decomposition import PCA

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df['Drug_Cluster'],
    palette='Set2',
    s=100
)

plt.title(
    "Drug Prescription Patterns Clustered Using K-Means",
    fontsize=14
)

plt.xlabel(
    "PCA Component 1"
)

plt.ylabel(
    "PCA Component 2"
)

plt.legend(
    title="Cluster"
)

plt.tight_layout()
plt.show()


# ============================================================
# OBJECTIVE 5:
# HIERARCHICAL CLUSTERING
# ============================================================

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)

X_cluster = StandardScaler().fit_transform(
    df[drug_features]
)

linked = linkage(
    X_cluster,
    method='ward'
)


plt.figure(figsize=(14, 6))

dendrogram(
    linked,
    truncate_mode='lastp',
    p=25,
    leaf_rotation=90,
    leaf_font_size=10,
    show_leaf_counts=True
)

plt.title(
    "Hierarchical Clustering Dendrogram for Regional Prescription Similarity"
)

plt.xlabel(
    "Clustered Regions"
)

plt.ylabel(
    "Euclidean Distance"
)

plt.tight_layout()
plt.show()

import warnings
warnings.filterwarnings('ignore')
import numpy as np   
import pandas as pd   
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_curve, auc, accuracy_score
)

# Load dataset
df = pd.read_csv(
    r"C:\Users\Varshitha\Downloads\Prescription_Monitoring_Program__Opioid__Benzodiazepine__Stimulant__Naloxone__and_Gabapentin_Prescription_Count___Rate_by_Patient_Residence.csv"
)

print("first five rows: \n", df.head())
print("Dataset Info: \n", df.info())
# Global Missing Value Handling
# Mean imputation is safe for numerical healthcare data
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
duplicates = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print("Number of duplicate rows removed:", duplicates)

print("Missing Values after cleaning:", df.isnull().sum().sum())
print("Statistical Summary: \n", df.describe())
# Feature Engineering 
df['Total Prescriptions'] = (
    df['Opiate Agonist Prescription Count'] +
    df['Benzodiazepine Prescription Count'] +
    df['Stimulant Prescription Count'] +
    df['Gabapentin Prescription Count']
)

df['High Naloxone Prescription'] = (
    df['Naloxone Prescription Count'] >
    df['Naloxone Prescription Count'].median()
).astype(int)
# OBJECTIVE 1:
# To analyze how prescriptions of different drugs are related.
drug_cols = [
    'Opiate Agonist Prescription Count',
    'Opiate Partial Agonist Prescription Count',
    'Benzodiazepine Prescription Count',
    'Stimulant Prescription Count',
    'Gabapentin Prescription Count',
    'Naloxone Prescription Count'
]

plt.figure(figsize=(14,10))
sns.heatmap(
    df[drug_cols].corr(),
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5
)
plt.title("Correlation Analysis Between Drug Prescription Types", fontsize=16)
plt.xticks(rotation=10, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
# OBJECTIVE 2
# Analyze Prescription Trends Over Time
# Polynomial Regression captures non-linear temporal patterns
df["Year"] = df["Date"].str.extract(r'(\d{4})').astype(int)
df["Quarter"] = df["Date"].str.extract(r'Quarter (\d)').astype(int)

df["Time_Index"] = df["Year"] + (df["Quarter"] - 1) / 4

trend = df.groupby("Time_Index")["Total Prescriptions"].sum().reset_index()

X = trend[["Time_Index"]]
y = trend["Total Prescriptions"]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

y_poly_pred = poly_model.predict(X_poly)

plt.figure(figsize=(9,5))
plt.scatter(X, y, label="Actual Prescriptions")
plt.plot(X, y_poly_pred, color="green", linewidth=2, label="Polynomial Trend (Degree 2)")
plt.xlabel("Time (Year + Quarter)")
plt.ylabel("Total Prescriptions")
plt.title("Prescription Trend Using Polynomial Regression")
plt.legend()
plt.tight_layout()
plt.show()

# Regression Evaluation Metrics
print("Polynomial Regression R²:", r2_score(y, y_poly_pred))
print("Polynomial Regression RMSE:",
      np.sqrt(mean_squared_error(y, y_poly_pred)))
# Objective 3:
#Objective : handle classification using KNN to predict high naloxone prescription risk
# K-Nearest Neighbors Classification
features = [
    'Opiate Agonist Prescription Count',
    'Benzodiazepine Prescription Count',
    'Stimulant Prescription Count',
    'Gabapentin Prescription Count',
    'Total Prescriptions'
]

X = df[features]
y = df['High Naloxone Prescription']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

plt.figure(figsize=(6,4))
sns.heatmap(
    confusion_matrix(y_test, knn_pred),
    annot=True,
    fmt='d',
    cmap='viridis'
)
plt.title("KNN Classification - Naloxone Risk Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Classification Report:\n", classification_report(y_test, knn_pred))
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))

# OBJECTIVE 4
# To build a Decision Tree model to generate interpretable rules
# explaining high naloxone prescription risk

from sklearn.tree import DecisionTreeClassifier, plot_tree

# Feature selection (same features, different purpose: rule extraction)
features = [
    'Opiate Agonist Prescription Count',
    'Benzodiazepine Prescription Count',
    'Stimulant Prescription Count',
    'Gabapentin Prescription Count',
    'Total Prescriptions'
]

X = df[features]
y = df['High Naloxone Prescription']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Decision Tree model
dt_model = DecisionTreeClassifier(
    max_depth=2,          # controls overfitting & improves readability
    min_samples_split=20,
    random_state=42
)

dt_model.fit(X_train, y_train)

# Predictions
dt_pred = dt_model.predict(X_test)

# Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Classification Report:\n",
      classification_report(y_test, dt_pred))
# Visualizing the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(
    dt_model,
    feature_names=features,
    class_names=['Low Risk', 'High Risk'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Rules for Naloxone Prescription Risk")
plt.show()

# OBJECTIVE 5
# Logistic Regression Model Evaluation
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("Accuracy :", accuracy_score(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Classification Performance Using Multiple Metrics")
plt.legend()
plt.tight_layout()
plt.show()

# Objective 6
# Compare Drug-Wise Prescription Distribution (K-Means)
from sklearn.cluster import KMeans

drug_features = [
    'Opiate Agonist Prescription Count',
    'Benzodiazepine Prescription Count',
    'Stimulant Prescription Count',
    'Gabapentin Prescription Count',
    'Naloxone Prescription Count'
]

X_drugs = df[drug_features]
X_scaled = StandardScaler().fit_transform(X_drugs)

wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df['Drug_Cluster'] = kmeans.fit_predict(X_scaled)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=df['Drug_Cluster'],
    palette='Set2',
    s=100
)
plt.title("Drug Prescription Patterns Clustered Using K-Means", fontsize=14)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# Objective 7
# Hierarchical Clustering
from scipy.cluster.hierarchy import dendrogram, linkage

X_cluster = StandardScaler().fit_transform(df[drug_features])
linked = linkage(X_cluster, method='ward')

plt.figure(figsize=(14,6))
dendrogram(
    linked,
    truncate_mode='lastp',
    p=25,
    leaf_rotation=90,
    leaf_font_size=10,
    show_leaf_counts=True
)
plt.title("Hierarchical Clustering Dendrogram for Regional Prescription Similarity")
plt.xlabel("Clustered Regions")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.show()

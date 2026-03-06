# =========================================
# 1. Import Libraries
# =========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

sns.set_style("whitegrid")

# =========================================
# 2. Load Dataset
# =========================================

df = pd.read_csv("Titanic-Dataset.csv")

print("First 5 rows:")
print(df.head())

# =========================================
# 3. Basic Information
# =========================================

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# =========================================
# 4. Check Missing Values
# =========================================

print("\nMissing Values:")
print(df.isnull().sum())

# =========================================
# 5. Exploratory Data Analysis (EDA)
# =========================================

# Survival distribution
plt.figure()
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Survival by gender
plt.figure()
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# Age distribution
plt.figure()
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Passenger class vs survival
plt.figure()
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Correlation heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# =========================================
# 6. Data Cleaning
# =========================================

# Fill missing Age with median
df["Age"].fillna(df["Age"].median(), inplace=True)

# Fill missing Embarked with most common value
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Drop Cabin because too many missing values
df.drop(columns=["Cabin"], inplace=True)

# =========================================
# 7. Encode Categorical Variables
# =========================================

# Convert Sex to numeric
df["Sex"] = df["Sex"].map({"male":0, "female":1})

# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# =========================================
# 8. Feature Selection
# =========================================

X = df.drop(columns=["Survived","Name","Ticket","PassengerId"])
y = df["Survived"]

print("\nFeatures used:")
print(X.head())

# =========================================
# 9. Feature Scaling
# =========================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Features:")
print(X_scaled.head())

# =========================================
# 10. Final Dataset Shape
# =========================================

print("\nFinal Dataset Shape:")

print(X_scaled.shape)

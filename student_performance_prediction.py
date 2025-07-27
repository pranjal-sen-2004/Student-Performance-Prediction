# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/StudentsPerformance.csv")

# Preview Data
print(df.head())

# Encode Categorical Columns
label_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
df_encoded = df.copy()

le = LabelEncoder()
for col in label_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Check for missing values
print("\nMissing values:\n", df_encoded.isnull().sum())

# Feature and Target
X = df_encoded.drop(['math score'], axis=1)  # Predicting math score
y = df_encoded['math score']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRoot Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted Math Score")
plt.grid(True)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df_encoded.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Feature Importance (from coefficients)
feature_importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance (from Linear Regression Coefficients):\n", feature_importance)

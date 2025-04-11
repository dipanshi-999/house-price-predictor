import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
df = pd.read_csv("train.csv")

# Drop high-missing columns
df = df.drop(['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu'], axis=1)

# Remove rows with missing target
df = df.dropna(subset=["SalePrice"])

# Features & labels
X = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']

# One-hot encode
X = pd.get_dummies(X)
X, y = X.align(y, axis=0, join='inner')

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fill missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and imputer
joblib.dump(model, 'house_price_model.pkl')
joblib.dump(imputer, 'imputer.pkl')

print("✅ Model and imputer saved successfully!")

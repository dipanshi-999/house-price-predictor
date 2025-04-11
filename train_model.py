import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# Load data
df = pd.read_csv("train.csv")

# Select only the features used in the app
X = df[["OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF"]]
y = df["SalePrice"]

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and imputer
joblib.dump(model, "house_price_model.pkl")
joblib.dump(imputer, "imputer.pkl")

print("✅ Model trained and saved successfully!")

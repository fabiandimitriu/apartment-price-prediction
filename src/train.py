import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# 1. Load data
data = pd.read_csv("data/raw/train.csv")


# 2. Select features
features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath"
]

X = data[features]
y = data["SalePrice"]


# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Train model
model = RandomForestRegressor()

model.fit(X_train, y_train)


# 5. Evaluate
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)

print("MAE:", mae)


# 6. Save model
joblib.dump(model, "models/model.joblib")

print("Model saved.")

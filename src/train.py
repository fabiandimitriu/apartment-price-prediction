import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Vârsta casei
    if "YearBuilt" in df.columns:
        df["HouseAge"] = 2025 - df["YearBuilt"]

    # Suprafață per cameră
    if "GrLivArea" in df.columns and "TotRmsAbvGrd" in df.columns:
        df["AreaPerRoom"] = df["GrLivArea"] / df["TotRmsAbvGrd"].replace(0, 1)

    # Scor total băi
    if "FullBath" in df.columns and "HalfBath" in df.columns:
        df["TotalBathScore"] = df["FullBath"] + 0.5 * df["HalfBath"]

    return df


def main():
    # 1. Load data
    data = pd.read_csv("data/raw/train.csv")

    # 2. Feature engineering
    data = add_engineered_features(data)

    # 3. Select features
    features = [
        "GrLivArea",
        "BedroomAbvGr",
        "FullBath",
        "YearBuilt",
        "GarageCars",
        "TotRmsAbvGrd",
        "HalfBath",
        "HouseAge",
        "AreaPerRoom",
        "TotalBathScore"
    ]

    # păstrăm doar coloanele care există în dataset
    features = [col for col in features if col in data.columns]

    X = data[features].copy()
    y = data["SalePrice"].copy()

    # tratăm eventualele valori lipsă
    X = X.fillna(X.median(numeric_only=True))

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Model with hyperparameters
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # 6. Train
    model.fit(X_train, y_train)

    # 7. Evaluate
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # 8. Save model package
    os.makedirs("models", exist_ok=True)

    model_package = {
        "model": model,
        "features": features
    }

    joblib.dump(model_package, "models/model.joblib")
    print("Model saved to models/model.joblib")

    # 9. Feature importance
    importances = pd.Series(model.feature_importances_, index=features)
    importances = importances.sort_values(ascending=False)

    print("\nFeature importance:")
    print(importances)

    os.makedirs("reports", exist_ok=True)

    plt.figure(figsize=(10, 6))
    importances.plot(kind="bar")
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png")
    print("Feature importance chart saved to reports/feature_importance.png")


if __name__ == "__main__":
    main()
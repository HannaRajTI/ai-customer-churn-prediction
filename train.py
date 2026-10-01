import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 2. DATA CLEANING
# ============================================================

# Convert TotalCharges from text to numeric.
# Invalid/blank values become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# customerID is an identifier, not a useful prediction feature.
df = df.drop("customerID", axis=1)

print("\nMissing values before preprocessing:")
print(df.isnull().sum())


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("Churn", axis=1)

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ============================================================
# 4. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# ============================================================
# 5. PREPROCESSING PIPELINES
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        # Fill missing numerical values with median
        ("imputer", SimpleImputer(strategy="median")),

        # Scale numerical features
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        # Fill missing categorical values
        ("imputer", SimpleImputer(strategy="most_frequent")),

        # Convert categories into numerical values
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)

print("\nTraining Logistic Regression...")

logistic_model.fit(X_train, y_train)


# ============================================================
# 8. RANDOM FOREST MODEL
# ============================================================

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

print("Training Random Forest...")

random_forest_model.fit(X_train, y_train)


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

def evaluate_model(model, name):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("\n" + name)
    print("-" * 40)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return f1


logistic_f1 = evaluate_model(
    logistic_model,
    "Logistic Regression"
)

random_forest_f1 = evaluate_model(
    random_forest_model,
    "Random Forest"
)


# ============================================================
# 10. SELECT MODEL
# ============================================================

if random_forest_f1 >= logistic_f1:

    best_model = random_forest_model
    best_model_name = "Random Forest"

else:

    best_model = logistic_model
    best_model_name = "Logistic Regression"


print("\nBest model:", best_model_name)


# ============================================================
# 11. SAVE MODEL
# ============================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/churn_model.pkl"
)

print("\nModel saved successfully.")
print("Location: models/churn_model.pkl")
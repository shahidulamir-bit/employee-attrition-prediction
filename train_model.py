from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

MODEL_DIR = Path(__file__).resolve().parent / "model"
DATA_FILE = MODEL_DIR / "employee_attrition_dataset.csv"
MODEL_FILE = MODEL_DIR / "employee_attrition_model.pkl"

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================
# 2. REMOVE UNNECESSARY COLUMN
# ==========================================

# Employee_ID is an identifier, not a useful
# feature for predicting employee attrition.

if "Employee_ID" in df.columns:
    df = df.drop(columns=["Employee_ID"])


# ==========================================
# 3. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Attrition"])

y = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ==========================================
# 6. RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 7. COMPLETE PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ==========================================
# 8. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ==========================================
# 10. MAKE PREDICTIONS
# ==========================================

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]


# ==========================================
# 11. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("Accuracy:", round(accuracy, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Attrition", "Attrition"]
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(
    pipeline,
    MODEL_FILE
)

print("\n================================")
print("MODEL SAVED")
print("================================")
print(
    f"Saved model as: {MODEL_FILE}"
)
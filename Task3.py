import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


# ============================
# Load loan dataset
# ============================
def load_loan_data(csv_path):
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Target variable
    y = df["default"]

    # Feature matrix
    X = df.drop(columns=["default"])

    return X, y


# ============================
# Train PD model
# ============================
def train_pd_model(csv_path):
    X, y = load_loan_data(csv_path)

    # Pipeline handles NaNs, scaling, and modeling
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    model.fit(X, y)

    return model, X.columns.tolist()


# ============================
# Predict probability of default
# ============================
def predict_pd(model, feature_names, loan_features):
    x = pd.DataFrame([loan_features], columns=feature_names)
    return float(model.predict_proba(x)[0][1])


# ============================
# Expected loss calculation
# ============================
def expected_loss(model, feature_names, loan_features, loan_amount, recovery_rate=0.10):
    pd_est = predict_pd(model, feature_names, loan_features)
    lgd = 1 - recovery_rate
    return pd_est * lgd * loan_amount


# ============================
# Run model
# ============================
if __name__ == "__main__":
    csv_path = "./Task 3 and 4_Loan_Data.csv"

    model, feature_names = train_pd_model(csv_path)

    print("Model trained using features:")
    print(feature_names)

    # Example borrower (adjust numbers freely)
    sample_loan = {
        "customer_id": 10001,
        "credit_lines_outstanding": 4,
        "loan_amt_outstanding": 15000,
        "total_debt_outstanding": 30000,
        "income": 85000,
        "years_employed": 6,
        "fico_score": 720
    }

    loan_amount = 250000

    pd_est = predict_pd(model, feature_names, sample_loan)
    el = expected_loss(model, feature_names, sample_loan, loan_amount)

    print("\nSample borrower PD:", round(pd_est, 4))
    print("Expected Loss ($):", round(el, 2))
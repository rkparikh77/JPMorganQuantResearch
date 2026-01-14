import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------------
# 1) Load data
# ----------------------------
def load_gas_data(csv_path: str) -> pd.DataFrame:
    """
    Expected columns: Dates, Prices
    Dates in format like 10/31/20, 1/31/21, ...
    """
    df = pd.read_csv(csv_path)
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)

    # Basic validation
    if "Prices" not in df.columns or "Dates" not in df.columns:
        raise ValueError("CSV must contain columns: Dates, Prices")

    return df


# ----------------------------
# 2) Interpolation for past dates (within observed range)
# ----------------------------
def build_interpolator(df: pd.DataFrame) -> pd.Series:
    """
    Build a daily time series from the monthly points and interpolate in time.
    This gives smooth estimates for any in-range date.
    """
    s = df.set_index("Dates")["Prices"].astype(float)

    # Make daily index and interpolate
    daily_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s_daily = s.reindex(daily_idx)
    s_daily = s_daily.interpolate(method="time")
    return s_daily


# ----------------------------
# 3) Extrapolation model (trend + seasonality)
# ----------------------------
def fit_trend_seasonality_model(df: pd.DataFrame):
    """
    Fit a lightweight regression model on monthly points:
    price(t) = a + b*t + seasonal terms

    We use day-based t and Fourier terms for yearly + semi-yearly seasonality.
    """
    dates = df["Dates"]
    y = df["Prices"].astype(float).to_numpy()

    t0 = dates.min()
    t_days = (dates - t0).dt.days.to_numpy().astype(float)

    # Seasonal features (annual + semi-annual)
    # Using 365.25 keeps it stable across years.
    w1 = 2.0 * np.pi / 365.25
    w2 = 2.0 * w1

    X = np.column_stack([
        np.ones_like(t_days),           # intercept
        t_days,                         # linear trend
        np.sin(w1 * t_days), np.cos(w1 * t_days),  # annual
        np.sin(w2 * t_days), np.cos(w2 * t_days),  # semi-annual
    ])

    # Least squares fit
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    def predict(date_like) -> float:
        d = pd.to_datetime(date_like)
        td = float((d - t0).days)

        x = np.array([
            1.0,
            td,
            np.sin(w1 * td), np.cos(w1 * td),
            np.sin(w2 * td), np.cos(w2 * td),
        ])

        val = float(x @ beta)
        # Natural gas prices should not be negative; clamp if model wiggles below 0.
        return max(0.0, val)

    return predict


# ----------------------------
# 4) Public API: estimate_price(date) -> float
# ----------------------------
def build_price_estimator(csv_path: str):
    df = load_gas_data(csv_path)

    s_daily = build_interpolator(df)
    model_predict = fit_trend_seasonality_model(df)

    min_date = df["Dates"].min()
    max_date = df["Dates"].max()
    max_forecast_date = max_date + pd.Timedelta(days=365)

    def estimate_price(date_like) -> float:
        d = pd.to_datetime(date_like)

        # In-range: use interpolation for best â€œgroundedâ€ estimate
        if min_date <= d <= max_date:
            return float(s_daily.loc[d.normalize()])

        # 1-year forecast range: use model
        if max_date < d <= max_forecast_date:
            return model_predict(d)

        # Outside allowed range: still return a model value, but you may choose to raise
        # depending on the simulation expectations.
        return model_predict(d)

    return estimate_price, df


# ----------------------------
# 5) Optional visualization
# ----------------------------
def plot_fit_and_forecast(df: pd.DataFrame, estimator, forecast_days: int = 365):
    import matplotlib
    matplotlib.use("Agg")  # << force non-GUI backend
    import matplotlib.pyplot as plt

    min_date = df["Dates"].min()
    max_date = df["Dates"].max()

    # Daily dates for plotting
    plot_end = max_date + pd.Timedelta(days=forecast_days)
    daily_dates = pd.date_range(min_date, plot_end, freq="D")
    y_hat = [estimator(d) for d in daily_dates]

    plt.figure()
    plt.plot(df["Dates"], df["Prices"], marker="o", linestyle="None", label="Monthly observed")
    plt.plot(daily_dates, y_hat, label="Estimated (interp + model forecast)")
    plt.axvline(max_date, linestyle="--", label="Last observed date")
    plt.title("Natural Gas Price: Interpolation + 1Y Extrapolation")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    plt.savefig("gas_price_forecast.png", dpi=150)
    plt.close()
    print("Saved plot to gas_price_forecast.png")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Update the path if needed
    CSV_PATH = "Nat_Gas.csv"

    estimator, df = build_price_estimator(CSV_PATH)

    # Example queries
    print("Example (in-range):", estimator("2022-03-15"))
    print("Example (forecast):", estimator(df["Dates"].max() + pd.Timedelta(days=180)))

    # Optional: visualize
    plot_fit_and_forecast(df, estimator, forecast_days=365)
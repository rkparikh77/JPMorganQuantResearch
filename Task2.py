import pandas as pd
import numpy as np
from typing import List
from datetime import datetime

# ============================================================
# 1) Load and prepare the monthly natural gas price data
# ============================================================

def load_gas_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)

    if "Prices" not in df.columns:
        raise ValueError("CSV must contain a Prices column")

    return df


# ============================================================
# 2) Interpolate monthly data into daily prices
# ============================================================

def build_interpolator(df: pd.DataFrame) -> pd.Series:
    s = df.set_index("Dates")["Prices"].astype(float)
    daily_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s_daily = s.reindex(daily_idx)
    return s_daily.interpolate(method="time")


# ============================================================
# 3) Fit extrapolation model (trend + seasonality)
# ============================================================

def fit_trend_seasonality_model(df: pd.DataFrame):
    dates = df["Dates"]
    prices = df["Prices"].astype(float).values

    t0 = dates.min()
    t = (dates - t0).dt.days.values

    w1 = 2 * np.pi / 365.25
    w2 = 2 * w1

    X = np.column_stack([
        np.ones_like(t),
        t,
        np.sin(w1 * t), np.cos(w1 * t),
        np.sin(w2 * t), np.cos(w2 * t)
    ])

    beta, *_ = np.linalg.lstsq(X, prices, rcond=None)

    def predict(date):
        d = pd.to_datetime(date)
        td = (d - t0).days
        x = np.array([
            1,
            td,
            np.sin(w1 * td), np.cos(w1 * td),
            np.sin(w2 * td), np.cos(w2 * td)
        ])
        price = x @ beta
        return max(price, 0.0)

    return predict


# ============================================================
# 4) Unified price estimator
# ============================================================

def build_price_estimator(csv_path: str):
    df = load_gas_data(csv_path)
    daily_prices = build_interpolator(df)
    model = fit_trend_seasonality_model(df)

    min_date = df["Dates"].min()
    max_date = df["Dates"].max()
    forecast_end = max_date + pd.Timedelta(days=365)

    def estimate_price(date):
        d = pd.to_datetime(date)

        if min_date <= d <= max_date:
            return float(daily_prices.loc[d])

        return model(d)

    return estimate_price, df


# ============================================================
# 5) Storage contract pricer
# ============================================================

def price_storage_contract(
    injection_dates: List[str],
    withdrawal_dates: List[str],
    injection_rate: float,
    withdrawal_rate: float,
    max_storage: float,
    storage_cost_per_day: float,
    volume_per_event: float,
    price_fn
):
    events = []

    for d in injection_dates:
        events.append((pd.to_datetime(d), "inject"))

    for d in withdrawal_dates:
        events.append((pd.to_datetime(d), "withdraw"))

    events.sort(key=lambda x: x[0])

    inventory = 0.0
    cashflow = 0.0
    last_date = events[0][0]

    for date, action in events:
        days = (date - last_date).days
        cashflow -= inventory * storage_cost_per_day * days

        price = price_fn(date)

        if action == "inject":
            qty = min(volume_per_event, injection_rate)
            if inventory + qty > max_storage:
                raise ValueError("Storage capacity exceeded")
            inventory += qty
            cashflow -= qty * price

        else:
            qty = min(volume_per_event, withdrawal_rate)
            if inventory < qty:
                raise ValueError("Not enough gas to withdraw")
            inventory -= qty
            cashflow += qty * price

        last_date = date

    return cashflow


# ============================================================
# 6) Plot (headless safe)
# ============================================================

def plot_fit_and_forecast(df, estimator, forecast_days=365):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    min_date = df["Dates"].min()
    max_date = df["Dates"].max()

    dates = pd.date_range(min_date, max_date + pd.Timedelta(days=forecast_days), freq="D")
    prices = [estimator(d) for d in dates]

    plt.figure()
    plt.plot(df["Dates"], df["Prices"], "o", label="Observed")
    plt.plot(dates, prices, label="Estimated")
    plt.axvline(max_date, linestyle="--", label="Forecast start")
    plt.legend()
    plt.savefig("gas_price_forecast.png")
    plt.close()


# ============================================================
# 7) Example run
# ============================================================

if __name__ == "__main__":
    estimator, df = build_price_estimator("Nat_Gas.csv")

    print("Example (in-range):", estimator("2022-03-15"))
    print("Example (forecast):", estimator(df["Dates"].max() + pd.Timedelta(days=180)))

    plot_fit_and_forecast(df, estimator)

    injections = ["2023-01-01", "2023-02-01", "2023-03-01"]
    withdrawals = ["2023-10-01", "2023-11-01", "2023-12-01"]

    value = price_storage_contract(
        injections,
        withdrawals,
        injection_rate=100_000,
        withdrawal_rate=100_000,
        max_storage=300_000,
        storage_cost_per_day=0.02,
        volume_per_event=100_000,
        price_fn=estimator
    )

    print("Storage contract value:", round(value, 2))
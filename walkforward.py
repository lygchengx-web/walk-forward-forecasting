import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = ["return_1", "return_5", "return_20", "volatility_20", "distance_20"]


def load_prices(raw, date_column, close_column):
    csv = pd.read_csv(io.BytesIO(raw), usecols=[date_column, close_column])
    dates = pd.to_datetime(csv[date_column], format="%Y-%m-%d", errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Dates must be present and unique.")
    closes = pd.to_numeric(csv[close_column].replace(".", np.nan), errors="raise")
    prices = pd.Series(closes.to_numpy(), index=dates, name="close").dropna().sort_index()
    if prices.empty or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Closing values must be finite and positive.")
    return prices


def demo_prices(seed=17):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.006, 1250)
    returns = noise.copy()
    for i in range(1, len(returns)):
        returns[i] += 0.35 * returns[i - 1]
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=pd.bdate_range("2019-01-02", periods=len(returns)))


def supervised(prices, horizon):
    log_price = np.log(prices)
    daily = log_price.diff()
    return pd.DataFrame(
        {
            "return_1": daily,
            "return_5": log_price.diff(5),
            "return_20": log_price.diff(20),
            "volatility_20": daily.rolling(20).std(),
            "distance_20": log_price - log_price.rolling(20).mean(),
            "target_date": prices.index.to_series().shift(-horizon),
            "realized_bps": (log_price.shift(-horizon) - log_price) * 10_000,
        }
    ).dropna()


def walk_forward(rows, folds, min_train):
    # Only labels whose end date is strictly before this test block are observable.
    if len(rows) < min_train:
        raise ValueError("Not enough rows for the training window.")
    targets = pd.DatetimeIndex(rows["target_date"])
    first_test = rows.index.searchsorted(targets[min_train - 1], side="right")
    if len(rows) - first_test < folds:
        raise ValueError("Not enough rows for the training window, label gap, and requested test folds.")
    for test in np.array_split(np.arange(first_test, len(rows)), folds):
        train_end = targets.searchsorted(rows.index[test[0]], side="left")
        yield slice(0, train_end), test


def errors(predictions):
    result = {}
    for model in ("flat", "ridge", "hist_gradient"):
        residual = predictions[model] - predictions["realized_bps"]
        result[model] = {
            "mae_bps": float(np.abs(residual).mean()),
            "rmse_bps": float(np.sqrt(np.square(residual).mean())),
        }
    return result


def evaluate(rows, folds=4, min_train=252):
    predictions = []
    windows = []
    for fold, (train, test) in enumerate(walk_forward(rows, folds, min_train), start=1):
        training = rows.iloc[train]
        testing = rows.iloc[test]
        result = testing[["target_date", "realized_bps"]].copy()
        result["fold"] = fold
        result["flat"] = 0.0
        models = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            "hist_gradient": HistGradientBoostingRegressor(
                max_iter=100, max_leaf_nodes=7, l2_regularization=10.0, early_stopping=False
            ),
        }
        for name, model in models.items():
            model.fit(training[FEATURES], training["realized_bps"])
            result[name] = model.predict(testing[FEATURES])
        predictions.append(result)
        windows.append(
            {
                "fold": fold,
                "train_rows": len(training),
                "last_train_label": training["target_date"].iloc[-1].date().isoformat(),
                "test_start": testing.index[0].date().isoformat(),
                "test_end": testing.index[-1].date().isoformat(),
                "test_rows": len(testing),
                "metrics": errors(result),
            }
        )
    combined = pd.concat(predictions).rename_axis("as_of")
    return combined, {"metrics": errors(combined), "folds": windows}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def main():
    parser = argparse.ArgumentParser(description="Compare return forecasts on expanding windows.")
    parser.add_argument("--csv", type=Path, help="Leave empty for the seeded synthetic example.")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--close-column", default="Close")
    parser.add_argument("--horizon", type=positive_int, default=1)
    parser.add_argument("--folds", type=positive_int, default=4)
    parser.add_argument("--min-train", type=positive_int, default=252)
    parser.add_argument("--out", type=Path, default=Path("run"))
    args = parser.parse_args()

    try:
        if args.csv:
            raw = args.csv.read_bytes()
            prices = load_prices(raw, args.date_column, args.close_column)
            source = {"kind": "csv", "sha256": hashlib.sha256(raw).hexdigest()}
        else:
            prices = demo_prices()
            source = {"kind": "synthetic", "seed": 17}
        predictions, summary = evaluate(supervised(prices, args.horizon), args.folds, args.min_train)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    summary.update(source=source, horizon=args.horizon, observations=len(prices), min_train=args.min_train)
    args.out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out / "predictions.csv")
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()

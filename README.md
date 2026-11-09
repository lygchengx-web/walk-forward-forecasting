# Walk-forward return forecasting

Can a few lagged returns beat a flat forecast? On the Fed's broad dollar index, they didn't. Ridge nearly matched the zero-return baseline. The boosted trees did worse in all four test blocks.

| Forecast | MAE (bp) | RMSE (bp) |
| --- | ---: | ---: |
| Zero return | **24.94** | **33.84** |
| Ridge | 25.07 | 34.01 |
| Histogram gradient boosting | 26.30 | 35.82 |

These are pooled errors on 3,985 out-of-sample one-observation-ahead forecasts. The input is the daily index from 2006 through 2024, with missing observations dropped. The test blocks run from December 2008 through December 2024. Hyperparameters are fixed before the walk-forward run.

Each row uses the closing value at its forecast origin and trailing returns and volatility. The label is the log return to the close `--horizon` observations later, expressed in basis points. At the start of a test block, training labels must end before the first test origin. Ridge's scaler is fitted inside each fold, and both models stay fixed through that fold. `metrics.json` records the fold boundaries and input file hash. `predictions.csv` keeps the dates, targets, and forecasts on the same rows.

## Run it

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python walkforward.py --out run/demo
```

The offline example has a deliberately autocorrelated price process. Ridge beats the flat forecast there, which is a check that the experiment can recover a planted signal.

```sh
curl -fsSL "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS&cosd=2006-01-01&coed=2024-12-31" -o dollar.csv
python walkforward.py --csv dollar.csv --date-column observation_date --close-column DTWEXBGS --min-train 756 --out run/dollar
```

For a local CSV with `Date,Close` columns, pass just `--csv`. Dates use YYYY-MM-DD. The benchmark evaluates forecast errors on one series and doesn't model positions, execution, or costs. FRED may revise the input history.

Data from the Board of Governors of the Federal Reserve System, [Nominal Broad U.S. Dollar Index (DTWEXBGS)](https://fred.stlouisfed.org/series/DTWEXBGS), via FRED. The table above was produced with Python 3.13, NumPy 2.1.3, pandas 2.2.3, and scikit-learn 1.6.1.

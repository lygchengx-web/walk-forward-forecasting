import unittest

import numpy as np
import pandas as pd

from walkforward import FEATURES, demo_prices, evaluate, load_prices, supervised, walk_forward


class ForecastTests(unittest.TestCase):
    def test_labels_are_aligned_and_gap_excludes_overlapping_targets(self):
        prices = demo_prices()
        rows = supervised(prices, 5)
        for train, test in walk_forward(rows, folds=4, min_train=252):
            origin = rows.index[test[0]]
            self.assertLess(rows.iloc[train]["target_date"].max(), origin)
            self.assertGreaterEqual(train.stop, 252)
            end = rows.loc[origin, "target_date"]
            self.assertEqual(prices.index.get_loc(end) - prices.index.get_loc(origin), 5)
            self.assertAlmostEqual(rows.loc[origin, "realized_bps"], np.log(prices[end] / prices[origin]) * 10_000)

    def test_future_values_cannot_change_existing_features(self):
        prices = demo_prices()
        cutoff = prices.index[400]
        altered = prices.copy()
        altered.loc[altered.index > cutoff] *= 9
        before = supervised(prices, 5).loc[:cutoff, FEATURES]
        after = supervised(altered, 5).loc[:cutoff, FEATURES]
        pd.testing.assert_frame_equal(before, after)

    def test_later_test_prices_cannot_change_first_prediction(self):
        prices = demo_prices().iloc[:280]
        result, _ = evaluate(supervised(prices, 1), folds=2, min_train=100)
        origin = result.index[0]
        prices.loc[prices.index > origin] *= 9
        changed, _ = evaluate(supervised(prices, 1), folds=2, min_train=100)
        np.testing.assert_allclose(result.loc[origin, ["ridge", "hist_gradient"]].astype(float),
                                   changed.loc[origin, ["ridge", "hist_gradient"]].astype(float))
        self.assertNotEqual(result.loc[origin, "realized_bps"], changed.loc[origin, "realized_bps"])
        self.assertTrue(result.index.is_unique)

    def test_loader_sorts_dates_and_rejects_duplicates(self):
        data = b"Date,Close\n2024-01-04,101\n2024-01-02,100\n2024-01-03,.\n"
        prices = load_prices(data, "Date", "Close")
        self.assertTrue(prices.index.is_monotonic_increasing)
        self.assertEqual(prices.tolist(), [100, 101])
        with self.assertRaisesRegex(ValueError, "unique"):
            load_prices(data + b"2024-01-02,100\n", "Date", "Close")

    def test_short_series_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "Not enough rows"):
            evaluate(supervised(demo_prices().iloc[:80], 5), folds=2, min_train=100)


if __name__ == "__main__":
    unittest.main()

# Comparing different cutoffs for calibrating the high-risk flag.
# The one with the best F1 is chosen for test.

import numpy as np
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from ml.config import CV_FOLDS, HIGH_RISK_PERCENTILE, RANDOM_STATE, TEST_SIZE
from ml.train import build_pipeline, calibrate_cutoff, flag_scores, load_data


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    truth_cut = float(np.percentile(y_train, HIGH_RISK_PERCENTILE))
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(build_pipeline(), X_train, y_train, cv=cv)
    actual = y_train >= truth_cut

    print(f"truth cutoff ({HIGH_RISK_PERCENTILE}th percentile of y_train): {truth_cut}")
    print("TRAINING split, out-of-fold predictions")
    print("cutoff  precision  recall  f1")
    for cut in [truth_cut, *range(120, 211, 10)]:
        p, r, f = flag_scores(oof, actual, cut)
        print(f"{cut:6.1f}  {p:9.4f}  {r:6.4f}  {f:6.4f}")

    best = calibrate_cutoff(X_train, y_train, truth_cut)
    p, r, f = flag_scores(oof, actual, best)
    print(f"Training with chosen cutoff {best}: precision={p:.4f} recall={r:.4f} f1={f:.4f}")

    pipe = build_pipeline().fit(X_train, y_train)
    p, r, f = flag_scores(pipe.predict(X_test), y_test >= truth_cut, best)
    print(f"Test split with chosen cutoff: precision={p:.4f} recall={r:.4f} f1={f:.4f}")


if __name__ == "__main__":
    main()

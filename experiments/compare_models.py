# Comparison to see if any other model improves performance

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import CV_FOLDS, RANDOM_STATE, TEST_SIZE
from ml.train import load_data


def main() -> None:
    X, y = load_data()
    X_train, _, y_train, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "LinearRegression (v0.1)": LinearRegression(),
        "Ridge alpha=1": Ridge(alpha=1.0),
        "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "LassoCV": LassoCV(cv=5, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    for name, model in candidates.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        scores = -cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error"
        )
        print(f"{name:26s} cv_rmse_mean={scores.mean():.4f}  cv_rmse_std={scores.std():.4f}")


if __name__ == "__main__":
    main()

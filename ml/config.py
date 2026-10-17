MODEL_VERSION = "v0.2"

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Model selection uses K-fold CV only for training
CV_FOLDS = 5
# Percentile of y_train to define high risk patients
HIGH_RISK_PERCENTILE = 75
# Cutoff calibrated in steps of this size
CUTOFF_STEP = 1.0
# Fixed API feature vector order
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

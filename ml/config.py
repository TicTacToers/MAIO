MODEL_VERSION = "v0.1"

RANDOM_STATE = 42
TEST_SIZE = 0.2
# Model selection uses K-fold CV on the training split; the test split is only reported.
CV_FOLDS = 5

# Keep order: the API builds the feature vector in this order.
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

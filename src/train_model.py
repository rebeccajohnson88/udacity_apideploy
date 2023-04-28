# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, compute_metrics_slice, create_calc_subset
import pandas as pd
import pickle
import logging


logging.basicConfig(filename = "../logs/modeling_log.log",
                    level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Add code to load in the data.
data = pd.read_csv("../data/census_clean.csv")
logger.info("read in data")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state = 99)
logger.info("generated train-test split")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
logger.info("processed training data")

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
     encoder = encoder, lb = lb
)
logger.info("processed test data")

# Train and save a model
model_result = train_model(X_train, y_train)
pickle.dump(model_result, open("../model/model_result.pkl", "wb"))
pickle.dump(encoder, open("../model/encoder.pkl", 'wb'))
pickle.dump(lb, open("../model/lb.pkl", 'wb'))
logging.info("Stored model result")

# Conduct inference
preds = inference(model_result, X_test)
logging.info("generated predictions")

# Evaluate results in overall data
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logging.info("Precision {prec}; recall {rec}; f beta {f}".format(prec = precision, 
        rec = recall, f = fbeta))

# Evaluate results separately for one categorical feature (education)
compute_metrics_slice(raw_data = test, one_feature = 'education',
                     y_test = y_test, preds = preds)
logging.info("Completed accuracy by subgroup calculation")

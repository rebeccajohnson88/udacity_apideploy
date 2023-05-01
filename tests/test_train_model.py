# Script to implement tests related to modeling functions

import pytest
import pandas as pd
import numpy as np
import logging
import pickle 
from ml.data import process_data
from ml.model import train_model, inference
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Set up logging
logging.basicConfig(filename = "./logs/testing_log.log",
                    level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Add fixtures for data and model objects
@pytest.fixture(name = "raw_data")
def data_loaded():
    data = pd.read_csv("../data/census_clean.csv")
    return data

@pytest.fixture(name = "model_result")
def model_loaded():
    with open('../model/model_result.pkl', 'rb') as f:
        model_result = pickle.load(f)
        f.close()
    return model_result


# Check whether the train test split results 
# in same dimensionality
def test_split(raw_data):
    train, test = train_test_split(raw_data, test_size=0.20, random_state = 99)
    try:
        assert train.shape[1] == test.shape[1]
    except AssertionError as err:
        logging.info("Different number of cols in train and test")
        raise err

# Check whether predictions are the correct type
def test_classifier(model_result):
    try:
        assert isinstance(model_result, LogisticRegression) 
    except AssertionError as err:
        logging.info("Model is not expected type")
        raise err

def test_predictions(model_result, raw_data):
    with open('../model/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
        f.close()
    with open('../model/lb.pkl', 'rb') as f:
        lb = pickle.load(f)
        f.close()
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
    train, test = train_test_split(raw_data, test_size=0.20, random_state = 99)
    X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
     encoder = encoder, lb = lb
    )
    preds = inference(model_result, X_test)
    try:
        assert isinstance(preds, np.ndarray)
    except AssertionError as err:
        logging.info("Predictions not expected type")
        raise err 

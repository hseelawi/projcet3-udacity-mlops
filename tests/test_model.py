import pandas as pd
import numpy as np
import pytest
from starter.ml.model import train_model, compute_model_metrics, inference, slice_performance
from starter.ml.data import process_data

@pytest.fixture
def data():
    data = "data/census_cleaned.csv"
    return pd.read_csv(data)

@pytest.fixture
def cat_features():
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
    return cat_features

@pytest.fixture
def data_processing(data, cat_features):
    X, y, _, _ = process_data(data, categorical_features=cat_features, label='salary')
    return X, y

@pytest.fixture
def model_training(data_processing):
    X, y = data_processing
    model = train_model(X, y)
    return model

def test_process_data(data, cat_features):
    # test if Shape of processed data is correct
    X, y, _, _ = process_data(data, categorical_features=cat_features, label='salary')
    assert X.shape[1] == 108

def test_train_model(data_processing):
    # test if model predictions have the correct shape
    X, y = data_processing
    model = train_model(X, y)
    assert model.predict(X).shape[0] == y.shape[0]
    
def test_inference(model_training, data_processing):
    # test if predictions are binary
    predictions = inference(model_training, data_processing[0])
    assert np.all(predictions < 2)

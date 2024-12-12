import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    save_model,
    load_model,
    compute_model_metrics,
    performance_on_categorical_slice,
)

@pytest.fixture
def sample_data():
    """ Fixture to create sample data for tests. """
    data = {
        "age": [39, 50],
        "workclass": ["State-gov", "Private"],
        "fnlgt": [77516, 83311],
        "education": ["Bachelors", "HS-grad"],
        "education-num": [13, 9],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Craft-repair"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", ">50K"]
    }
    return pd.DataFrame(data)

def test_train_model(sample_data):
    categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    label = "salary"
    X, y, _, _ = process_data(sample_data, categorical_features, label, training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_inference(sample_data):
    categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    label = "salary"
    X, y, _, _ = process_data(sample_data, categorical_features, label, training=True)
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)

def test_save_and_load_model(sample_data, tmp_path):
    categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    label = "salary"
    X, y, _, _ = process_data(sample_data, categorical_features, label, training=True)
    model = train_model(X, y)
    file_path = tmp_path / "test_model.pkl"
    save_model(model, file_path)
    loaded_model = load_model(file_path)
    preds = inference(loaded_model, X)
    assert len(preds) == len(y)

def test_performance_on_categorical_slice(sample_data):
    categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    label = "salary"
    X, y, encoder, lb = process_data(sample_data, categorical_features, label, training=True)
    model = train_model(X, y)
    precision, recall, fbeta = performance_on_categorical_slice(
        sample_data,
        column_name="workclass",
        slice_value="State-gov",
        categorical_features=categorical_features,
        label=label,
        encoder=encoder,
        lb=lb,
        model=model
    )
    assert precision >= 0 and recall >= 0 and fbeta >= 0

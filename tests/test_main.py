import json

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_get_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"greeting":"Hello World!"} 


def test_predict_pos():
    data = {'age': 50, 'workclass': 'Private', 'fnlgt': 367260, 
    'education': 'Bachelors', 'education-num': 13, 
    'marital-status': 'Never-married', 'occupation': 'Tech-support',
    'relationship': 'Unmarried', 'race': 'White', 'sex': 'Male',
    'capital-gain': 14084, 'capital-loss': 0, 'hours-per-week': 45,
    'native-country': 'Canada'}
    resp = client.post("/predict", data=json.dumps(data))
    assert resp.status_code == 200
    assert resp.json()["prediction"] == '1'

def test_predict_neg():
    data = {'age': 18, 'workclass': 'Private', 'fnlgt': 367260,
    'education': 'Bachelors', 'education-num': 13,
    'marital-status': 'Never-married', 'occupation': 'Tech-support',
    'relationship': 'Unmarried', 'race': 'White', 'sex': 'Male',
    'capital-gain': 14084, 'capital-loss': 0, 'hours-per-week': 0,
    'native-country': 'Canada'}
    resp = client.post("/predict", data=json.dumps(data))
    assert resp.status_code == 200
    assert resp.json()["prediction"] == '0'



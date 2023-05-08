# Script to implement tests related to API call

from fastapi.testclient import TestClient
import json
from main import app

## initialize client
client = TestClient(app)

## test get call
def test_welcome_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to this app! You'll predict salary categories using demographic features"

## test case > 50k
def test_predict_over50():

    data_topost = {
            'age': 47,
            'workclass': 'Private',
            'fnlgt': 51835,
            'education': 'Prof-school',
            'education_num': 15,
            'marital_status': 'Married-civ-spouse',
            'occupation': 'Prof-specialty',
            'relationship': 'Wife',
            'race': 'White',
            'sex': 'Female',
            'capital_gain': 0,
            'capital_loss': 1902,
            'hours_per_week': 60,
            'native_country': 'Honduras'
            }
    data_topost_json = json.dumps(data_topost)
    response = client.post("/predictions", data=data_topost_json)

    # test response code and output
    assert response.status_code == 200
    assert response.json()["predicted_salary"] == '>50K'



## test case <= 50k
def test_predict_under50():

    data_topost = {
        'age': 25,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 176756,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Never-married',
        'occupation': 'Farming-fishing',
        'relationship': 'Own-child',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 35,
        'native-country': 'United-States'}

    data_topost_json = json.dumps(data_topost)
    response = client.post("/predictions", data=data_topost_json)

    # test response code and output
    assert response.status_code == 200
    assert response.json()["predicted_salary"] == '<=50K'


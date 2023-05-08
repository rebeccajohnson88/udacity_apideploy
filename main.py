# Code for creating the API

## Imports
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import pandas as pd

from src.ml.data import process_data
from src.ml.model import inference
import pickle

import numpy as np

## Define categorical features
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

# Define InputData class
class InputData(BaseModel):
	age: int
	workclass: str
	fnlgt: int
	education: str
	education_num: int
	marital_status: int
	occupation: str
	relationship: str
	race: str
	sex: str
	capital_gain: int
	capital_loss: int
	hours_per_week: int
	native_country: str

	class Config:
		schema_extra = {
    		"example": {
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
  		}


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def welcome_page():
    return "Welcome to this app! You'll predict salary categories using demographic features"

# Define a POST for model inference 
@app.post("/predictions")
async def predict_results(input_data: InputData):

	# Turn input into dataframe
	raw_data = pd.DataFrame([dict(input_data)])

	# Load model objects
	with open('model/model_result.pkl', 'rb') as f:
		model_result = pickle.load(f)
		f.close()
	with open('model/encoder.pkl', 'rb') as f:
		encoder = pickle.load(f)
		f.close()
	
	with open('model/lb.pkl', 'rb') as f:
		lb = pickle.load(f)
		f.close()


    # Process data 
	X_new, y_new, encoder_new, lb_new = process_data(
    raw_data, categorical_features=cat_features, training=False,
     encoder = encoder, lb = lb)

    # Generate inferences 
	pred_raw = inference(model_result, X_new)
	pred_label = ">50K" if pred_raw[0] > 0.5 else "<=50K"

    # Return prediction
	return({'predicted_salary': pred_label})


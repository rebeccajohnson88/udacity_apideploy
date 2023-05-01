# Code for creating the API

## Imports
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def welcome_page():
    return "Welcome to this app! You'll predict salary categories using demographic features"

# Define a POST for model inference 
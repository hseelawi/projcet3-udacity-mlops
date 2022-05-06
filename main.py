import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from joblib import load
from starter.ml.data import process_data
from starter.ml.model import inference
import os

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

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

model = load('./model/classifier.joblib')
encoder = load('./model/encoder.joblib')

class RequestData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": "42",
                "workclass": "Private",
                "fnlgt": 413297,
                "education": "Some-college",
                "education-num": 10,
                "marital-status": "Married-civ-spouse",
                "occupation": "Tech-support",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

@app.post("/predict")
def predict(request: RequestData):
    request_df = pd.DataFrame.from_dict([request.dict(by_alias=True)])
    X, _, _, _ = process_data(request_df, categorical_features=cat_features, training=False, encoder=encoder)
    predicted_label = inference(model, X)
    return {'prediction': str(predicted_label[0])}

if __name__ == "__main__":
    uvicorn.run(app)

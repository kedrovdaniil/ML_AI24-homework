import io
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List

import pandas as pd

from starlette.responses import StreamingResponse


class Car(BaseModel):
    name: str
    year: int
    # selling_price: Optional[int]
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Cars(BaseModel):
    objects: List[Car]

class CarResponse(BaseModel):
    prediction: float

def get_predict(cars: List[Car]) -> dict:
    with open("elasticnet_pipeline.joblib", 'rb') as file:
        pipeline = joblib.load(file)

    return pd.DataFrame(pipeline.predict(cars), columns=['prediction'])

def preprocessing(cars: List[Car]) -> dict:
    df = pd.DataFrame([vars(car) for car in cars], columns=Car.model_fields)
    print(df.head(), df.dtypes)
    return df

def extract_data_from_csv_file(contents: bytes) -> pd.DataFrame:
    stream = io.StringIO(contents.decode("utf-8"))
    df = pd.read_csv(stream)
    return df

def write_data_to_csv_file(data: pd.DataFrame):
    stream = io.StringIO()
    data.to_csv(stream, index=False)
    stream.seek(0) # Переместить указатель потока в начало

    response = StreamingResponse(stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"

    return response

ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    ml_models["cars_elastic_net"] = get_predict
    yield
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.post("/predict_item")
def predict_item(car: Car) -> CarResponse:
    try:
        Car.model_validate(car)
    except ValidationError as e:
        return {"error": str(e)}

    df = preprocessing([car])

    response = ml_models["cars_elastic_net"](df)
    print('RESPONSE: ', response['prediction'].iloc[0])
    return {'prediction': response['prediction'].iloc[0]}

@app.post("/predict_items")
def predict_items(file: UploadFile):
    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    finally:
        file.file.close()

    try:
        df = extract_data_from_csv_file(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {e}")

    try:
        response = ml_models["cars_elastic_net"](df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {e}")

    return write_data_to_csv_file(response)
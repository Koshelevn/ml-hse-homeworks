import pickle
import csv
import codecs
import io
import pandas as pd
import numpy as np

from typing import List, Optional, Tuple
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

from utils.helpers import df_processor

app = FastAPI()

artifacts = {}


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float
    engine: float
    max_power: float
    torque: float
    seats: float
    max_torque_rpm: float


class Items(BaseModel):
    objects: List[Item]


def make_predicts(items: List[Item], return_df: bool = False) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """
    Predict prices for cars.

    :param items: List of Item objects with cars data
    :param return_df: True if need to return base dataset
    :return:
    """
    base_data = pd.DataFrame(list(map(lambda x: dict(x), items)), columns=list(Item.model_fields.keys()))
    data = df_processor(base_data.copy())
    cat = ['brand', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    data_transformed = artifacts['encoder'].transform(data[cat])
    data_transformed = pd.DataFrame.sparse.from_spmatrix(data_transformed,
                                                         columns=artifacts['encoder'].get_feature_names_out())
    data = pd.concat([data_transformed, data], axis=1).drop(cat, axis=1)
    data = artifacts['scaler'].transform(data)
    predicts = np.exp(artifacts['model'].predict(data))
    if return_df:
        return predicts, base_data
    return predicts


@app.on_event("startup")
def startup_event():
    """Load pickle files on service startup"""

    artifacts["model"] = pickle.load(open('artifacts/best_model.pkl', 'rb'))
    artifacts["scaler"] = pickle.load(open('artifacts/scaler.pkl', 'rb'))
    artifacts["encoder"] = pickle.load(open('artifacts/encoder.pkl', 'rb'))


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Predict price for a singe car.

    :param item: dict with Item params
    :return: car value
    """
    predict = make_predicts([item])
    return np.round(predict, 2)


@app.post("/predict_items")
def predict_csv(file: UploadFile) -> StreamingResponse:
    """
    Predict car prices for csv of cars.

    :param file: csv file with cars data
    :return: csv file from response + price column added
    """
    csv_reader = csv.DictReader(codecs.iterdecode(file.file, 'utf-8'))
    items = Items(objects=[Item(**params) for params in list(csv_reader)])
    predicts, df = make_predicts(items.objects, True)
    df['price'] = np.round(predicts, 2)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=response.csv"
    return response

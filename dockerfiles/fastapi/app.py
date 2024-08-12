import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "crime_prediction_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Esquema de entrada para el modelo de predicción de robos en CABA según datos del clima.

    Esta clase define los campos de entrada así como una descripción de cada uno y sus límites de validación.

    :param is_day: Indica si es día o noche. 0: Noche. 1: Día.
    :param sunshine_duration. Duración de la luz solar, en segundos, para la hora dada (0 a 3600).
    :param temperature_2m: Temperatura del aire a 2 m del suelo, en °C (-20 a 50).
    :param apparent_temperature: Temperatura aparente, en °C (-20 a 50).
    :param wind_gusts: Rafagas de viento, en km/h. (0 a 100).
    :param dew_point_2m: Temperatura del punto de rocío a 2 m el suelo, en °C (-20 a 50).
    :param cloud_cover: Cobertura total de nubes como fracción de área, en % (0,100).
    :param cloud_cover_low: Cobertura de nubes y niebla hasta 2km de altitud, en % (0,100).
    :param cloud_cover_mid: Cobertura de nubes y niebla desde 2km de altitud hasta 6km, en % (0,100).
    :param cloud_cover_high: Cobertura de nubes y niebla mayor a 6km de altitud, en % (0,100). 
    :param wind_speed_10m: Velocidad del viento a 10m del suelo, en km/h (0 a 100).
    :param weather_code: Condición climática como código numérico.
    :param rain: Cantidad de precipitaciones del hora anterior, en mm (0 a 100).
    :param precipitation: Suma de precipitaciones totales de la hora anterior, en mm (0 a 100).
    :param wind_direction_10m: Dirección del viento a 10m sobre el suelo, en ° (0 a 360).
    :param surface_pressure: Presión en la superficie, en hPa (500 a 1500).
    :param pressure_msl: Presión del aire atmosférico a nivel del mar, en hPa (500 a 1500).
    :param relative_humidity_2m: Humedad relativa a 2m del suelo, en % (0 a 100).
    """

    is_day: int = Field(
        description="Indica si es día o noche. 0: Noche. 1: Día",
        ge=0,
        le=1,
    )
    sunshine_duration: float = Field(
        description="Duración de la luz solar, en segundos",
        ge=0,
        le=3600,
    )
    temperature_2m: float = Field(
        description="Temperatura del aire a 2 m del suelo, en °C",
        ge=-20,
        le=50,
    )
    apparent_temperature: float = Field(
        description="Temperatura aparente, en °C",
        ge=-20,
        le=50,
    )
    wind_gusts: float = Field(
        description="Ráfagas de viento, en km/h",
        ge=0,
        le=100,
    )
    dew_point_2m: float = Field(
        description="Temperatura del punto de rocío a 2 m el suelo, en °C",
        ge=-20,
        le=50,
    )
    cloud_cover: float = Field(
        description="Cobertura total de nubes como fracción de área, en %",
        ge=0,
        le=100,
    )
    cloud_cover_low: float = Field(
        description="Cobertura de nubes y niebla hasta 2km de altitud, en %",
        ge=0,
        le=100,
    )
    cloud_cover_mid: float = Field(
        description="Cobertura de nubes y niebla desde 2km de altitud hasta 6km, en %",
        ge=0,
        le=100,
    )
    cloud_cover_high: float = Field(
        description="Cobertura de nubes y niebla mayor a 6km de altitud, en %",
        ge=0,
        le=100,
    )
    wind_speed_10m: float = Field(
        description="Velocidad del viento a 10m del suelo, en km/h",
        ge=0,
        le=100,
    )
    weather_code: int = Field(
        description="Condición climática como código numérico",
    )
    rain: float = Field(
        description="Cantidad de precipitaciones de la hora anterior, en mm",
        ge=0,
        le=100,
    )
    precipitation: float = Field(
        description="Suma de precipitaciones totales de la hora anterior, en mm",
        ge=0,
        le=100,
    )
    wind_direction_10m: float = Field(
        description="Dirección del viento a 10m sobre el suelo, en °",
        ge=0,
        le=360,
    )
    surface_pressure: float = Field(
        description="Presión en la superficie, en hPa",
        ge=500,
        le=1500,
    )
    pressure_msl: float = Field(
        description="Presión del aire atmosférico a nivel del mar, en hPa",
        ge=500,
        le=1500,
    )
    relative_humidity_2m: float = Field(
        description="Humedad relativa a 2m del suelo, en %",
        ge=0,
        le=100,
    )

    model_config = {
    "json_schema_extra": {
        "examples":
            [
                {
                    "is_day": 1,
                    "sunshine_duration": 980,
                    "temperature_2m": 25.0,
                    "apparent_temperature": 27.0,
                    "wind_gusts": 45.0,
                    "dew_point_2m": 15.0,
                    "cloud_cover": 50,
                    "cloud_cover_low": 30,
                    "cloud_cover_mid": 20,
                    "cloud_cover_high": 10,
                    "wind_speed_10m": 20.0,
                    "weather_code": 1,
                    "rain": 0.0,
                    "precipitation": 0.0,
                    "wind_direction_10m": 154.0,
                    "surface_pressure": 1015.0,
                    "pressure_msl": 1010.0,
                    "relative_humidity_2m": 65,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Esquema de entrada para el modelo de predicción de robos en CABA según datos del clima.

    Esta clase define los campos de salida así como una descripción de cada uno y sus límites de validación.

    :param float_output: Salida del modelo. Número que indica la cantidad de robos para las condiciones climáticas dadas.
    :param str_output: Salida del modelo en formato de cadena de caracteres.
    """

    float_output: float = Field(
        description="Salida del modelo en formato numérico. Indica cuántos robos se predicen para el día y hora dados, según condiciones climáticas.",
    )
    str_output: str = Field(
        description="Salda del modelo en formato string.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "float_output": 3.5,
                    "str_output": "Según las condiciones climáticas de entrada se predice que hoy habrán 3.5 robos.",
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model("crime_prediction_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint de la API para predecir robos en CABA según condiciones climáticas.

    Este endpoint retorna un JSON con un mensaje para comprobar que la API funciona correctamente.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Bienvenido a la API para predecir robos en CABA según condiciones climáticas."}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint para predecir cantidad de robos según condiciones climáticas.

    Recibe como features las condiciones climáticas del día y hora en el que se quiere hacer la predicción.
    """

    # Extrae los features del request en un formato diccionario y los convierte a listas
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Ahora es posible crear un dataframe con los features
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # Ordenamos el dataframe para que las columnas tengan el mismo orden que espera recibir el modelo
    features_df = features_df[data_dict["features_column_order"]]

    # Aplicamos estandarización (standard scaler)
    features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    # Se hace la predicción utilizando el mejor modelo entrenado
    prediction = model.predict(features_df)

    # Obtengo salida en float y en string
    float_output = prediction[0]
    str_output = f"Se predicen {float_output:.1f} robos según las condiciones climáticas dadas."

    # Se lanza un task para chequear asincrónicamente si el modelo ha cambiado
    background_tasks.add_task(check_model)

    # Retorna resultado de la predicción
    return ModelOutput(float_output=float_output, str_output=str_output)

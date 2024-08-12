import datetime

from airflow.decorators import dag, task

markdown_text = """
### Proceso ETL para datos de clima y delitos en la Ciudad de Buenos Aires

- Este DAG extrae la información de los archivos CSV correspondientes. 
- Preprocesa la data para armar un único dataset.
- Finalmente guarda los sets de entrenamiento y validación en un bucket de S3.
"""


default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="etl_process",
    description="Proceso ETL para datos de clima y delitos en la Ciudad de Buenos Aires.",
    doc_md=markdown_text,
    tags=["ETL"],
    default_args=default_args,
    catchup=False,
)
def etl_process():

    @task.virtualenv(
        task_id="get_weather_data",
        requirements=["pandas==1.5.0",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def get_weather_data():
        """
        Carga el csv de clima y lo preprocesa
        """
        import awswrangler as wr
        import pandas as pd
        from airflow.models import Variable

        # Leemos el csv
        clima_df = pd.read_csv(Variable.get('weather_csv_url'))

        # Hacemos una copia del dataset para preprocesar
        clima_df_clean = clima_df.copy()

        # Convertimos a formato datetime
        clima_df_clean['date'] = pd.to_datetime(clima_df_clean['date'])

        # Quitamos la zona horaria
        clima_df_clean['date'] = clima_df_clean['date'].dt.tz_localize(None)

        # No necesitamos los registros de clima de 2015, 2023 y 2024, ya que no contamos con información de los delitos en esos años
        years_to_remove = [2015, 2023, 2024]
        clima_df_clean = clima_df_clean[~clima_df_clean['date'].dt.year.isin(years_to_remove)]

        # Eliminamos columnas innecesarias
        clima_df.drop(columns=['temperature_2m', 'precipitation', 'pressure_msl', 'wind_gusts_10m'], inplace=True)

        # Guardamos el dataset en el bucket
        data_path = "s3://data/clima.csv"
        wr.s3.to_csv(df=clima_df_clean,
                     path=data_path,
                     index=False)
        
        return data_path
        
    @task.virtualenv(
        task_id="get_crime_data",
        requirements=["pandas==1.5.0",
                      "awswrangler==3.6.0"],
        system_site_packages=True
    )
    def get_crime_data():
        """
        Carga el csv de delitos y lo preprocesa
        """
        import awswrangler as wr
        import pandas as pd
        from airflow.models import Variable

        # Leemos el csv
        delitos_df = pd.read_csv(Variable.get('crime_csv_url'))
        
        # Siendo la categoría con más cantidad de ocurrencias, nos enfocaremos en la predicción de robos totales únicamente
        robos_df = delitos_df[delitos_df['tipo'] == "Robo"]
        robos_df = robos_df[robos_df['subtipo'] == "Robo total"]

        # Hacemos una copia del dataset para preprocesar
        robos_df_clean = robos_df.copy()

        # Eliminamos las filas que no tienen fecha o franja
        robos_df_clean = robos_df_clean.dropna(subset=['fecha', 'franja'])

        # Convertimos la columna de fecha al formato datetime
        robos_df_clean['fecha'] = pd.to_datetime(robos_df_clean['fecha'], format='%Y-%m-%d')

        # Convertimos la columna de franja a string y le damos el formato correcto
        robos_df_clean['franja'] = robos_df_clean['franja'].astype(int).astype(str).str.zfill(2) + ':00:00'

        # Combinamos fecha y franja en una sola columna
        robos_df_clean['date'] = pd.to_datetime(robos_df_clean['fecha'].astype(str) + ' ' + robos_df_clean['franja'], format='%Y-%m-%d %H:%M:%S')

        # Eliminamos columnas innecesarias
        robos_df_clean.drop(columns=['id-mapa', 'anio', 'mes', 'dia', 'fecha', 'franja', 'barrio', 'comuna', 'latitud', 'longitud', 'tipo', 'subtipo', 'uso_arma', 'uso_moto'], inplace=True)

        # Agrupamos por fecha y hora, sumando la cantidad de robos
        robos_df_clean = robos_df_clean.groupby('date', as_index=False).agg({'cantidad': 'sum'})

        # Guardamos el dataset en el bucket
        data_path = "s3://data/delitos.csv"
        wr.s3.to_csv(df=robos_df_clean,
                     path=data_path,
                     index=False)
        
        return data_path
        
    @task.virtualenv(
        task_id="join_datasets",
        requirements=["pandas==1.5.0",
                      "awswrangler==3.6.0",
                      "s3fs",
                      "numpy"],
        system_site_packages=True
    )
    def join_datasets(clima_path, delitos_path):
        """
        Joinea ambos datasets por fecha y hora
        """
        import awswrangler as wr
        import pandas as pd
        import numpy as np

        clima_df = pd.read_csv(clima_path)
        delitos_df = pd.read_csv(delitos_path)

        # Hacemos un left join para mantener los datos de clima en el caso en que no haya delitos para un día y hora específicos
        dataset_final = pd.merge(clima_df, delitos_df, on='date', how='left')

        # En el caso que haya valores vacíos para la cantidad de delitos, asignamos un cero
        dataset_final['cantidad'] = dataset_final['cantidad'].fillna(0)

        # Renombramos la columna que corresponde a la cantidad de robos
        dataset_final.rename(columns={'cantidad': 'cantidad_robos'}, inplace=True)

        # Una vez realizado el join, ya no necesitamos la columna de fecha
        dataset_final.drop(columns=['date'], inplace=True)

        # Aplico feature engineering, modificando el target por el logaritmo
        dataset_final["cantidad_robos_log"] = np.log(dataset_final["cantidad_robos"]+1)

        # Guardamos el dataset en el bucket
        data_path = "s3://data/dataset-final.csv"
        wr.s3.to_csv(df=dataset_final,
                     path=data_path,
                     index=False)
        
        return data_path
    
    @task.virtualenv(
        task_id="split_dataset",
        requirements=["pandas==1.5.0",
                      "awswrangler==3.6.0",
                      "scikit-learn==1.3.2",
                      "s3fs",
                      "boto3"],
        system_site_packages=True
    )
    def split_dataset(df_path):
        """
        Separamos el dataset en 70% train y 30% test
        """
        import awswrangler as wr
        import boto3
        import json
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        dataset_final = pd.read_csv(df_path)

        X = dataset_final.drop(columns=['cantidad_robos','cantidad_robos_log'])
        y = dataset_final['cantidad_robos_log']

        # Se separa el dataset en entrenamiento y evaluación
        X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size=0.3, random_state=42)

        # Escalemos los datos
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = scaler.fit_transform(X_train)
        X_test_arr = scaler.transform(X_test)
        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        # Guardamos los datos del scaler dentro del bucket data de s3, en data/data_info/data.json
        client = boto3.client('s3')
        features_column_order = X.columns
        data_dict = {
                'columns' : dataset_final.drop(columns=['cantidad_robos']).columns.tolist(),
                'features_column_order' : features_column_order.tolist(),
                'standard_scaler_mean' : scaler.mean_.tolist(),
                'standard_scaler_std' : scaler.scale_.tolist(),
            }
        data_string = json.dumps(data_dict, indent=2)

        try:
            client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
            )
        except:
            pass

        # Guardamos los datasets en el bucket
        save_to_csv(X_train, "s3://data/train/X_train.csv")
        save_to_csv(X_test, "s3://data/test/X_test.csv")
        save_to_csv(y_train, "s3://data/train/y_train.csv")
        save_to_csv(y_test, "s3://data/test/y_test.csv")

        dataset = {
            "train": X_train,
            "test": X_test
        }

        return dataset

    @task.virtualenv(
        task_id="track_in_mlflow",
        requirements=["mlflow==2.10.2"],
        system_site_packages=True
    )
    def register_in_mlflow(dataset):
        """
        Registramos el experimento en MLflow
        """
        import mlflow

        mlflow.set_tracking_uri('http://mlflow:5000')
        
        experiment_name = "etl_process"

        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(name=experiment_name) 

        experiment = mlflow.get_experiment_by_name(experiment_name)

        with mlflow.start_run(experiment_id = experiment.experiment_id):
            mlflow.log_param("Train observations", dataset['train'].shape[0])
            mlflow.log_param("Test observations", dataset['test'].shape[0])

    path_weather = get_weather_data()
    path_crime = get_crime_data()
    path_joined_df = join_datasets(path_weather, path_crime)
    dataset = split_dataset(path_joined_df)
    register_in_mlflow(dataset)


dag = etl_process()
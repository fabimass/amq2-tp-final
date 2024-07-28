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

        # Guardamos el dataset en el bucket
        data_path = "s3://data/clima.csv"
        wr.s3.to_csv(df=clima_df_clean,
                     path=data_path,
                     index=False)
        
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


    get_weather_data() >> get_crime_data()


dag = etl_process()
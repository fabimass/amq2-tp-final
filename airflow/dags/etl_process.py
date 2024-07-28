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
        requirements=["pandas==2.2.2",
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

        df = pd.read_csv(f"https://drive.google.com/uc?id={Variable.get('weather_csv_id')}")

        data_path = "s3://data/clima.csv"

        wr.s3.to_csv(df=df,
                     path=data_path,
                     index=False)
        
    @task.virtualenv(
        task_id="get_crime_data",
        requirements=["pandas==2.2.2",
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

        df = pd.read_csv(f"https://drive.google.com/uc?id={Variable.get('crime_csv_id')}")

        data_path = "s3://data/delitos.csv"

        wr.s3.to_csv(df=df,
                     path=data_path,
                     index=False)


    get_weather_data() >> get_crime_data()


dag = etl_process()
import streamlit as st
import requests

st.title('Predictor de robos')

st.write("""
Ingresar datos de las condiciones climáticas
""")

with st.form(key='inputForm'):

    is_day = st.checkbox('Es de día?', value=True)
    sunshine_duration = st.number_input('Duración de la luz solar, en segundos (0 a 3600)', value=980)
    temperature_2m = st.number_input('Temperatura del aire, °C (-20 a 50)', value=25.0)
    apparent_temperature = st.number_input('Temperatura aparente, °C (-20 a 50)', value=27.0)
    wind_gusts_10m = st.number_input('Rafagas de viento, km/h. (0 a 100)', value=45.0)
    dew_point_2m = st.number_input('Temperatura del punto de rocío, °C (-20 a 50)', value=15.0)
    cloud_cover = st.number_input('obertura total de nubes como fracción de área, en % (0,100)', value=50)
    cloud_cover_low = st.number_input('Cobertura de nubes y niebla hasta 2km de altitud, en % (0,100)', value=30)
    cloud_cover_mid = st.number_input('Cobertura de nubes y niebla desde 2km de altitud hasta 6km, en % (0,100)', value=20)
    cloud_cover_high = st.number_input('Cobertura de nubes y niebla mayor a 6km de altitud, en % (0,100)', value=10)
    wind_speed_10m = st.number_input('Velocidad del viento a 10m del suelo, en km/h (0 a 100)', value=20.0)
    weather_code = st.number_input('Condición climática como código numérico', value=1)
    rain = st.number_input('Cantidad de precipitaciones, en mm (0 a 100)', value=0.0)
    precipitation = st.number_input('Suma de precipitaciones totales, en mm (0 a 100)', value=0.0)
    wind_direction_10m = st.number_input('Dirección del viento a 10m sobre el suelo, en ° (0 a 360)', value=154.0)
    surface_pressure = st.number_input('Presión en la superficie, en hPa (500 a 1500)', value=1015.0)
    pressure_msl = st.number_input('Presión del aire atmosférico a nivel del mar, en hPa (500 a 1500)', value=1010.0)
    relative_humidity_2m = st.number_input('Humedad relativa a 2m del suelo, en % (0 a 100)', value=65)

    submit_btn = st.form_submit_button(label='Enviar')

if submit_btn:

    url='http://fastapi:8800/predict'

    payload = {
        'features':{
            'is_day': is_day,
            'sunshine_duration': sunshine_duration,
            'temperature_2m': temperature_2m,
            'apparent_temperature': apparent_temperature,
            'wind_gusts_10m': wind_gusts_10m,
            'dew_point_2m': dew_point_2m,
            'cloud_cover': cloud_cover,
            'cloud_cover_low': cloud_cover_low,
            'cloud_cover_mid': cloud_cover_mid,
            'cloud_cover_high': cloud_cover_high,
            'wind_speed_10m': wind_speed_10m,
            'weather_code': weather_code,
            'rain': rain,
            'precipitation': precipitation,
            'wind_direction_10m': wind_direction_10m,
            'surface_pressure': surface_pressure,
            'pressure_msl': pressure_msl,
            'relative_humidity_2m': relative_humidity_2m
        }
    }

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, json=payload, headers=headers)

        res = response.json()
        st.success(f'{res["str_output"]}')

        # Muestro lo que devolvio la peticion
        with st.expander(f'Respuesta de FastAPI'):

            st.write(f'**Codigo de estado:** {response.status_code}')

            st.write(f'**Respuesta del servidor**')
            st.json(res)

    except requests.exceptions.RequestException as e:
        st.error(f'Error al realizar la peticion: {e}')

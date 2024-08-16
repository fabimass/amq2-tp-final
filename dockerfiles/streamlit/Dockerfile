FROM python:3.8-slim

WORKDIR /app
COPY . .
COPY requirements.txt /requirements.txt

RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /requirements.txt

EXPOSE 8501

CMD ["streamlit","run","frontend.py","--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rest_api.py .
COPY model/ model/
COPY data/zipcode_demographics.csv data/

EXPOSE 8000

CMD ["python", "rest_api.py"]

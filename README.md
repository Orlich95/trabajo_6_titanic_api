# TP6 API Titanic

API en FastAPI para servir un modelo de predicción del Titanic.

## Endpoints

- `GET /`
- `GET /health`
- `POST /predict`

## Ejecución local

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t titanic-api:v1.0 .
docker run -d -p 8000:8000 --name titanic-api titanic-api:v1.0
```
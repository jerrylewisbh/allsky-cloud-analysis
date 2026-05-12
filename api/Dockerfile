FROM python:3.11-slim

WORKDIR /app
RUN pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

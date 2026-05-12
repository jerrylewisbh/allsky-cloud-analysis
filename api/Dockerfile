FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic pyephem

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

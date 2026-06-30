# Thin layer on the official MLflow image adding the Postgres driver, which the
# upstream image does not ship. Keeps everything else identical to upstream.
FROM ghcr.io/mlflow/mlflow:v3.14.0

RUN pip install --no-cache-dir psycopg2-binary

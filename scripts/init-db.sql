-- Creates a dedicated `mlflow` role + database on first boot of the postgres
-- container. lakeFS keeps its own `lakefs` db (created by POSTGRES_DB). This
-- keeps MLflow tracking data in the same postgres instance, separate schema/db.
CREATE ROLE mlflow WITH LOGIN PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

# Stage 1: Builder (установка зависимостей)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Установи системные зависимости
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создай виртуальное окружение
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Обнови pip и установи uv (аналог poetry для синхронизации зависимостей)
RUN pip install --upgrade pip && pip install uv

# Скопируй зависимости (requirements.txt или pyproject.toml)
COPY requirements.txt /app/requirements.txt
RUN uv pip install -r /app/requirements.txt

# Stage 2: Runtime (минимальный образ для запуска)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Установи минимальные системные зависимости
RUN apt-get update && apt-get install -y \
    python3.11 \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Скопируй venv из builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Создай директории для MLflow
RUN mkdir -p /app/mlruns

# Скопируй код проекта
COPY . /app
WORKDIR /app

# Настрой переменные окружения для MLflow
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns

# Открой порт для MLflow UI
EXPOSE 5000

# Команда по умолчанию: запуск MLflow сервера
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////app/mlruns/mlflow.db", "--default-artifact-root", "/app/mlruns", "--host", "0.0.0.0", "--port", "5000"]
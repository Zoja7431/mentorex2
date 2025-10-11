# Stage 1: Builder (установка зависимостей)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

# Установи системные зависимости
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создай виртуальное окружение
RUN python3.11 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Обнови pip с большим timeout и зеркалом (если нужно)
RUN pip install --upgrade pip --timeout 100 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Скопируй requirements.txt и установи пакеты с timeout
COPY requirements.txt /app/requirements.txt
# Установи PyTorch с CUDA поддержкой из официального индекса
RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --timeout 100
# Установи остальные пакеты из китайского зеркала
RUN pip install -r /app/requirements.txt --timeout 100 --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Stage 2: Runtime (минимальный образ для запуска)
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Установи минимальные системные зависимости
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    git \
    sqlite3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создай пользователя без root прав
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Скопируй venv из builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Создай директории для MLflow
RUN mkdir -p /app/mlruns && chown -R appuser:appuser /app

# Скопируй код проекта
COPY . /app
WORKDIR /app

# Настрой переменные окружения для MLflow
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns

# Смени пользователя
USER appuser

# Открой порт для MLflow UI
EXPOSE 5000

# Добавь health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Команда по умолчанию: запуск MLflow сервера
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:////app/mlruns/mlflow.db", "--default-artifact-root", "/app/mlruns", "--host", "0.0.0.0", "--port", "5000"]
# mentorex2
# здесь опишу способ открытия проекта через wsl 
    wsl 
    cd /mnt/d/mentorex2

# запуск окружения 
    source .venv/bin/activate
    dvc --version
    git pull 
    dvc pull --force

# запус mlflow сервера
    mkdir -p ~/mentorex2/mlruns
    mlflow server --backend-store-uri sqlite:////home/mentorex/mentorex2/mlruns/mlflow.db --default-artifact-root /home/mentorex/mentorex2/mlruns --host 0.0.0.0 --port 5000 &

# перезапуск окружения
    rm -rf /mnt/d/mentorex2/.venv
    python3 -m venv /mnt/d/mentorex2/.venv
    source /mnt/d/mentorex2/.venv/bin/activate
    uv sync
    
# Commit (Git + DVC): После каждого успешного dvc repro или изменения (новая модель/данные). Шаги:
    dvc repro # обновляет dvc.lock (хеши outs).
    git add dvc.lock dvc.yaml .dvc # коммить метаданные.
    git commit -m "Trained BERT, added metrics".
    git push
    dvc push






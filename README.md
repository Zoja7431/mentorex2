# 🚀 Mentorex2 - MLOps Project

> **Comprehensive Machine Learning Pipeline with Experiment Tracking, Model Versioning, and Containerization**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.4.0-orange.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-green.svg)](https://dvc.org)

## Project Overview

This project demonstrates a complete MLOps pipeline implementing multiple machine learning models with proper experiment tracking, data versioning, and containerization. The project includes:

- **5 Different Model Types**: BERT, ViT, CNN, RNN (LSTM/GRU), and Boosting (XGBoost, LightGBM, CatBoost)
- **2 Datasets**: IMDB Movie Reviews (sentiment analysis) and CIFAR-10 (image classification)
- **Full MLOps Stack**: DVC for data versioning, MLflow for experiment tracking, Docker for containerization

## Architecture

```
mentorex2/
├── 📁 mentorex2/           # Main package
│   ├── 📁 data/            # Data storage (raw, interim, processed)
│   ├── 📁 models/          # Trained models
│   ├── 📁 modeling/        # Training and prediction scripts
│   ├── 📁 reports/         # Generated reports and plots
│   └── 📁 logs/            # Training logs
├── 📁 notebooks/           # Jupyter notebooks for experimentation
├── 🐳 Dockerfile           # Container configuration
├── 📄 dvc.yaml            # DVC pipeline definition
├── 📄 requirements.txt    # Python dependencies
└── 📄 pyproject.toml      # Project configuration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker
- Git
- DVC

### 1. Clone and Setup
```bash
git clone https://github.com/Zoja7431/mentorex2.git
cd mentorex2
```

### 2. Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# OR using uv (faster)
uv sync
```

### 3. Data Setup
```bash
# Pull data using DVC
dvc pull --force
```

### 4. Run MLflow Server
```bash
mkdir -p mlruns
mlflow server --backend-store-uri sqlite:///mlruns/mlflow.db \
              --default-artifact-root mlruns \
              --host 0.0.0.0 --port 5000
```

### 5. Run Pipeline
```bash
# Process data
dvc repro process_imdb
dvc repro process_cifar10_vit
dvc repro process_cifar10_cnn

# Train models
dvc repro train_bert
dvc repro train_rnn
dvc repro train_boosting
dvc repro train_vit
dvc repro train_cnn

# Generate predictions and plots
dvc repro predict
dvc repro generate_plots
```

## 🐳 Docker Usage

### Build Image
```bash
docker build -t mentorex2-mlflow .
```

### Run Container
```bash
# Run MLflow server
docker run -p 5000:5000 mentorex2-mlflow

# Run training
docker run --gpus all mentorex2-mlflow python mentorex2/mentorex2/modeling/train.py --model bert
```

## 📊 Models & Performance

| Model | Dataset | Task | Best Accuracy | Status |
|-------|---------|------|---------------|--------|
| BERT | IMDB | Sentiment Analysis | 84.87% | ✅ Trained |
| LSTM | IMDB | Sentiment Analysis | ~82% | ✅ Trained |
| GRU | IMDB | Sentiment Analysis | ~81% | ✅ Trained |
| XGBoost | IMDB | Sentiment Analysis | ~78% | ✅ Trained |
| LightGBM | IMDB | Sentiment Analysis | ~77% | ✅ Trained |
| CatBoost | IMDB | Sentiment Analysis | ~76% | ✅ Trained |
| ViT | CIFAR-10 | Image Classification | ~85% | ✅ Trained |
| CNN | CIFAR-10 | Image Classification | ~82% | ✅ Trained |

## 🔧 Configuration

All hyperparameters and paths are centralized in `mentorex2/mentorex2/config.py`:

- **BERT**: 3 epochs, lr=2e-5, batch_size=16
- **RNN**: 10 epochs, lr=5e-4, hidden_dim=128
- **ViT**: 5 epochs, lr=1e-4, batch_size=64
- **CNN**: 50 epochs, lr=1e-3, batch_size=64

## 📈 Experiment Tracking

All experiments are tracked in MLflow with:
- **Metrics**: Training/validation loss, accuracy
- **Parameters**: Hyperparameters, model configurations
- **Artifacts**: Model files, metrics JSON, plots
- **Experiments**: Separate experiments for each model type

Access MLflow UI at: `http://localhost:5000`

## 🛠️ MLOps Features

### ✅ Implemented
- [x] **Data Versioning** with DVC
- [x] **Experiment Tracking** with MLflow
- [x] **Model Versioning** and storage
- [x] **Containerization** with Docker
- [x] **Pipeline Automation** with DVC
- [x] **Logging** and monitoring
- [x] **Multi-model Support** (5 different architectures)
- [x] **GPU Support** with CUDA

## 📁 Data Structure

```
data/
├── raw/                    # Original datasets
│   ├── IMDB Dataset.csv
│   └── cifar10/
├── interim/               # Intermediate processing results
└── processed/             # Final processed data for training
```

## Key Features

1. **Multi-Model Pipeline**: Supports both NLP and Computer Vision tasks
2. **Reproducible Experiments**: All experiments are fully reproducible
3. **Scalable Architecture**: Easy to add new models and datasets
4. **Production Ready**: Docker containerization for deployment
5. **Comprehensive Logging**: Detailed logs for debugging and monitoring

## 📚 Usage Examples

### Train a specific model
```bash
python mentorex2/mentorex2/modeling/train.py --model bert
python mentorex2/mentorex2/modeling/train.py --model vit
```

### Make predictions
```bash
python mentorex2/mentorex2/modeling/predict.py
```

### Generate plots
```bash
python mentorex2/mentorex2/plots.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Author

**Zoja7431** - [GitHub](https://github.com/Zoja7431)

---

⭐ **Star this repository if you found it helpful!**

import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import ViTForImageClassification, BertForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from torch.utils.data import TensorDataset, DataLoader
import catboost as cb
import pickle
import json
import mlflow
import logging
import argparse
from tqdm import tqdm
from mentorex2.mentorex2.config import (
    NUM_CLASSES_CIFAR, EPOCHS_VIT, EPOCHS_CNN, LEARNING_RATE_VIT, LEARNING_RATE_CNN, WEIGHT_DECAY, LABEL_SMOOTHING,
    EPOCHS_BERT, LEARNING_RATE_BERT, EPOCHS_RNN, LEARNING_RATE_RNN, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
    DROPOUT, XGBOOST_PARAM_GRID, LIGHTGBM_PARAM_GRID, CATBOOST_PARAM_GRID, OUTPUT_DIR_VIT, OUTPUT_DIR_CNN, OUTPUT_DIR_BERT,
    OUTPUT_DIR_RNN, OUTPUT_DIR_BOOSTING, BATCH_SIZE_RNN, PROCESSED_DIR, LOGS_DIR
)

# Настройка MLflow Tracking URI
mlflow.set_tracking_uri("http://0.0.0.0:5000")  # Указываем адрес твоего MLflow сервера
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Настройка логирования
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, 'train.log'))
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {torch.version.cuda}")
else:
    logger.warning("CUDA is not available, training on CPU")


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(256 * 8 * 8, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, rnn_type='LSTM'):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn_type = rnn_type
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        embeds = self.embedding(x)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=True)
        if self.rnn_type == 'LSTM':
            packed_out, (h_n, c_n) = self.rnn(packed_embeds)
        else:
            packed_out, h_n = self.rnn(packed_embeds)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        out = self.dropout(h_n)
        return self.fc(out)

def train_vit(train_loader, test_loader, output_dir):
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=NUM_CLASSES_CIFAR,
        ignore_mismatched_sizes=True
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_VIT, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Настройка эксперимента MLflow
    experiment_name = "mentorex2_vit"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    training_stats = []
    with mlflow.start_run(run_name="vit_training", experiment_id=experiment_id):
        for epoch in range(EPOCHS_VIT):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"ViT Epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            avg_train_loss = running_loss / len(train_loader)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).logits
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            avg_test_accuracy = 100 * correct / total

            training_stats.append({
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Accur.': avg_test_accuracy
            })

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", avg_test_accuracy, step=epoch)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(training_stats, f)
            
        mlflow.log_artifact(os.path.join(output_dir, 'metrics.json'))
        # Фикс: Добавь input_example (первый батч из train_loader)
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            sample_input = sample_batch[0][:1]  # Один сэмпл для примера
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="vit_model",
            input_example=sample_input,
            registered_model_name="mentorex2_vit"  # Регистрирует в Model Registry
        )

        model.save_pretrained(output_dir)  # Для ViT
    return model, training_stats

def train_cnn(train_loader, test_loader, output_dir):
    model = SimpleCNN(num_classes=NUM_CLASSES_CIFAR).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_CNN, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Настройка эксперимента MLflow
    experiment_name = "mentorex2_cnn"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    training_stats = []
    with mlflow.start_run(run_name="cnn_training", experiment_id=experiment_id):
        for epoch in range(EPOCHS_CNN):
            model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f"CNN Epoch {epoch+1}"):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            avg_train_loss = running_loss / len(train_loader)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            avg_test_accuracy = 100 * correct / total

            training_stats.append({
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Accur.': avg_test_accuracy
            })

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", avg_test_accuracy, step=epoch)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(training_stats, f)
        mlflow.log_artifact(os.path.join(output_dir, 'metrics.json'))
        # Фикс: Добавь input_example (первый батч из train_loader)
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            sample_input = sample_batch[0][:1]  # Один сэмпл для примера
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="cnn_model",
            input_example=sample_input,
            registered_model_name="mentorex2_cnn"  # Регистрирует в Model Registry
        )

        mlflow.pytorch.log_model(model, "cnn_model")

        torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
        mlflow.log_artifact(os.path.join(output_dir, 'cnn_model.pth'))
    return model, training_stats

def train_bert(train_loader, test_loader, output_dir):
    logger.info("Starting BERT training...")
    try:
        # Убедимся, что device определяется один раз
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}, CUDA Version: {torch.version.cuda}")
            torch.cuda.empty_cache()
        else:
            logger.warning("CUDA is not available, training on CPU")
        # Загрузка данных на CPU
        logger.info("Loading IMDB data for BERT...")
        data_files = [
            'imdb_train_input_ids.pt', 'imdb_train_attention_masks.pt', 'imdb_train_labels_bert.pt',
            'imdb_test_input_ids.pt', 'imdb_test_attention_masks.pt', 'imdb_test_labels_bert.pt'
        ]
        for file in data_files:
            file_path = os.path.join(PROCESSED_DIR, file)
            if not os.path.exists(file_path):
                logger.error(f"Data file {file_path} does not exist")
                raise FileNotFoundError(f"Data file {file_path} does not exist")
        
        train_input_ids = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'), map_location=device, weights_only=True)
        train_attention_masks = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'), map_location=device, weights_only=True)
        train_labels = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'), map_location=device, weights_only=True)
        test_input_ids = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'), map_location=device, weights_only=True)
        test_attention_masks = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'), map_location=device, weights_only=True)
        test_labels = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'), map_location=device, weights_only=True)
        logger.info("IMDB data loaded successfully")

        # Проверка формата данных
        logger.info(f"Train input_ids shape: {train_input_ids.shape}, dtype: {train_input_ids.dtype}, device: {train_input_ids.device}")
        logger.info(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}, device: {train_labels.device}")

        # Создание датасетов
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)
        logger.info(f"Created DataLoaders: train batches={len(train_loader)}, test batches={len(test_loader)}")

        # Инициализация модели
        logger.info("Initializing BERT model...")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
        logger.info(f"BERT model moved to {device}, Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_BERT, weight_decay=0.01)

        # Настройка эксперимента MLflow
        experiment_name = "mentorex2_bert"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Обучение
        training_stats = []
        with mlflow.start_run(run_name="bert_training", experiment_id=experiment_id):
            for epoch in range(EPOCHS_BERT):
                model.train()
                total_train_loss = 0
                for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"BERT Epoch {epoch+1}")):
                    b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    logger.debug(f"Batch {batch_idx}: input_ids device: {b_input_ids.device}, labels device: {b_labels.device}")
                    model.zero_grad()
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs.loss
                    total_train_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    logger.debug(f"Batch {batch_idx}: Loss = {loss.item():.4f}, GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

                avg_train_loss = total_train_loss / len(train_loader)
                logger.info(f"BERT Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

                model.eval()
                total_eval_accuracy = 0
                total_eval_loss = 0
                for batch in test_loader:
                    b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    with torch.no_grad():
                        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs.loss
                    logits = outputs.logits
                    total_eval_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    total_eval_accuracy += torch.mean((predictions == b_labels).float()).item()

                avg_val_loss = total_eval_loss / len(test_loader)
                avg_val_accuracy = total_eval_accuracy / len(test_loader)
                logger.info(f"BERT Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}, Val Accuracy = {avg_val_accuracy:.4f}")

                training_stats.append({
                    'epoch': epoch + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy
                })

                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", avg_val_accuracy, step=epoch)

            # Сохранение модели и метрик
            os.makedirs(output_dir, exist_ok=True)
            metrics_path = os.path.join(output_dir, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(training_stats, f)
            mlflow.log_artifact(metrics_path)
            mlflow.pytorch.log_model(model, "bert_model")
            logger.info(f"Saving BERT model to {output_dir}")
            model.save_pretrained(output_dir)
            logger.info(f"BERT training completed, metrics saved to {metrics_path}, GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

        return model, training_stats
    except Exception as e:
        logger.error(f"Error in train_bert: {e}")
        raise

def train_rnn(train_data, test_data, output_dir, rnn_type='LSTM'):
    train_loader = torch.utils.data.DataLoader(train_data, sampler=torch.utils.data.RandomSampler(train_data), batch_size=BATCH_SIZE_RNN)
    test_loader = torch.utils.data.DataLoader(test_data, sampler=torch.utils.data.SequentialSampler(test_data), batch_size=BATCH_SIZE_RNN)

    model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, rnn_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_RNN, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Настройка эксперимента MLflow
    experiment_name = "mentorex2_rnn"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    training_stats = []
    with mlflow.start_run(run_name=f"rnn_training", experiment_id=experiment_id):
        for epoch in range(EPOCHS_RNN):
            model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader, desc=f"{rnn_type} Epoch {epoch+1}"):
                inputs, lengths, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                sorted_idx = torch.argsort(lengths, descending=True)
                inputs, lengths, labels = inputs[sorted_idx], lengths[sorted_idx], labels[sorted_idx]
                optimizer.zero_grad()
                outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)

            model.eval()
            total_eval_loss = 0
            total_eval_acc = 0
            for batch in test_loader:
                inputs, lengths, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                sorted_idx = torch.argsort(lengths, descending=True)
                inputs, lengths, labels = inputs[sorted_idx], lengths[sorted_idx], labels[sorted_idx]
                with torch.no_grad():
                    outputs = model(inputs, lengths)
                loss = criterion(outputs, labels)
                total_eval_loss += loss.item()
                total_eval_acc += np.mean(torch.argmax(outputs, dim=1).cpu().numpy() == labels.cpu().numpy())

            avg_val_loss = total_eval_loss / len(test_loader)
            avg_val_accuracy = total_eval_acc / len(test_loader)

            training_stats.append({
                'epoch': epoch + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy
            })

            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", avg_val_accuracy, step=epoch)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{rnn_type.lower()}_metrics.pkl'), 'wb') as f:
            pickle.dump(training_stats, f)
        mlflow.log_artifact(os.path.join(output_dir, f'{rnn_type.lower()}_metrics.pkl'))
        mlflow.pytorch.log_model(model, f"{rnn_type.lower()}_model")

        torch.save(model.state_dict(), os.path.join(output_dir, f'{rnn_type.lower()}_model.pth'))
    return model, training_stats

def train_boosting(X_train, y_train, X_test, y_test, output_dir):
    models = {
        'XGBoost': (xgb.XGBClassifier(eval_metric='logloss'), XGBOOST_PARAM_GRID),
        'LightGBM': (lgb.LGBMClassifier(), LIGHTGBM_PARAM_GRID),
        'CatBoost': (cb.CatBoostClassifier(verbose=0, train_dir=os.path.join(output_dir, 'catboost_info')), CATBOOST_PARAM_GRID)
    }

    # Настройка эксперимента MLflow
    experiment_name = "mentorex2_boosting"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    results = {}
    with mlflow.start_run(run_name="mentorex2_boosting", experiment_id=experiment_id):
        for name, (model, param_grid) in models.items():
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {
                'model': best_model,
                'accuracy': acc,
                'val_scores': grid_search.cv_results_['mean_test_score'],
                'epochs': np.arange(1, len(grid_search.cv_results_['mean_test_score']) + 1)
            }
            with mlflow.start_run(nested=True):  # Nested run для модели
                mlflow.log_param(f"{name}_params", grid_search.best_params_)
                mlflow.log_metric(f"{name}_accuracy", acc)
                for i, score in enumerate(results[name]['val_scores']):
                    mlflow.log_metric(f"{name}_val_score", score, step=i)
                os.makedirs(output_dir, exist_ok=True)
                model_path = os.path.join(output_dir, f'{name.lower()}_model')
                if name == 'XGBoost':
                    best_model.save_model(f"{model_path}.json")
                    mlflow.xgboost.log_model(best_model, f"{name}_model")
                    mlflow.log_artifact(f"{model_path}.json")
                elif name == 'LightGBM':
                    with open(os.path.join(output_dir, 'lightgbm_model.pkl'), 'wb') as f:
                        pickle.dump(best_model, f)
                    mlflow.lightgbm.log_model(best_model, f"{name}_model")
                    mlflow.log_artifact(f"{model_path}.pkl")
                else:
                    best_model.save_model(f"{model_path}.cbm")
                    mlflow.catboost.log_model(best_model, f"{name}_model")
                    mlflow.log_artifact(f"{model_path}.cbm")

        # Глобальные метрики/results
        with open(os.path.join(output_dir, 'boosting_metrics.pkl'), 'wb') as f:
            pickle.dump(results, f)
        mlflow.log_artifact(os.path.join(output_dir, 'boosting_metrics.pkl'))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for mentorex2 project")
    parser.add_argument("--model", type=str, required=True, choices=["vit", "cnn", "bert", "rnn", "boosting"],
                        help="Model to train: vit, cnn, bert, rnn, or boosting")
    args = parser.parse_args()

    if args.model == "vit":
        logger.info("Loading CIFAR-10 data for ViT...")
        train_images = np.load(os.path.join(PROCESSED_DIR, 'cifar10_train_images_vit.npy'),  weights_only=True)
        train_labels = np.load(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_vit.npy'),  weights_only=True)
        test_images = np.load(os.path.join(PROCESSED_DIR, 'cifar10_test_images_vit.npy'),  weights_only=True)
        test_labels = np.load(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_vit.npy'),  weights_only=True)

        # Преобразование в тензоры (ViT ожидает NHWC -> NCHW)
        train_images = torch.from_numpy(train_images).float().permute(0, 3, 1, 2)
        train_labels = torch.from_numpy(train_labels).long()
        test_images = torch.from_numpy(test_images).float().permute(0, 3, 1, 2)
        test_labels = torch.from_numpy(test_labels).long()

        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_CIFAR, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_CIFAR, shuffle=False, num_workers=4, pin_memory=True)

        logger.info("Starting ViT training...")
        train_vit(train_loader, test_loader, OUTPUT_DIR_VIT)

    elif args.model == "cnn":
        logger.info("Loading CIFAR-10 data for CNN...")
        train_images = np.load(os.path.join(PROCESSED_DIR, 'cifar10_train_images_cnn.npy'),  weights_only=True)
        train_labels = np.load(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_cnn.npy'),  weights_only=True)
        test_images = np.load(os.path.join(PROCESSED_DIR, 'cifar10_test_images_cnn.npy'),  weights_only=True)
        test_labels = np.load(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_cnn.npy'),  weights_only=True)

        # Преобразование в тензоры (CNN ожидает NCHW)
        train_images = torch.from_numpy(train_images).float().permute(0, 3, 1, 2)
        train_labels = torch.from_numpy(train_labels).long()
        test_images = torch.from_numpy(test_images).float().permute(0, 3, 1, 2)
        test_labels = torch.from_numpy(test_labels).long()

        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_CIFAR, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_CIFAR, shuffle=False, num_workers=4, pin_memory=True)

        logger.info("Starting CNN training...")
        train_cnn(train_loader, test_loader, OUTPUT_DIR_CNN)

    elif args.model == "bert":
        logger.info("Starting BERT training...")
        # Функция train_bert игнорирует входные loaders, так что передаем None
        train_bert(None, None, OUTPUT_DIR_BERT)

    elif args.model == "rnn":
        logger.info("Loading IMDB data for RNN...")
        train_padded = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_padded_rnn.pt'),  weights_only=True)
        train_lengths = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_lengths_rnn.pt'),  weights_only=True)
        train_labels = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_rnn.pt'),  weights_only=True)
        test_padded = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_padded_rnn.pt'),  weights_only=True)
        test_lengths = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_lengths_rnn.pt'),  weights_only=True)
        test_labels = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_rnn.pt'),  weights_only=True)

        train_data = TensorDataset(train_padded, train_lengths, train_labels)
        test_data = TensorDataset(test_padded, test_lengths, test_labels)

        logger.info("Starting LSTM training...")
        train_rnn(train_data, test_data, OUTPUT_DIR_RNN, rnn_type='LSTM')

        logger.info("Starting GRU training...")
        train_rnn(train_data, test_data, OUTPUT_DIR_RNN, rnn_type='GRU')

    elif args.model == "boosting":
        logger.info("Loading IMDB data for Boosting...")
        with open(os.path.join(PROCESSED_DIR, 'imdb_train_tfidf.pkl'), 'rb') as f:
            X_train = pickle.load(f)
        with open(os.path.join(PROCESSED_DIR, 'imdb_test_tfidf.pkl'), 'rb') as f:
            X_test = pickle.load(f)
        y_train = np.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_boosting.npy'))
        y_test = np.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_boosting.npy'))

        logger.info("Starting Boosting training...")
        train_boosting(X_train, y_train, X_test, y_test, OUTPUT_DIR_BOOSTING)
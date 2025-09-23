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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


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

    training_stats = []
    with mlflow.start_run(run_name="vit_training"):
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
        mlflow.pytorch.log_model(model, "vit_model")

        model.save_pretrained(output_dir)
    return model, training_stats

def train_cnn(train_loader, test_loader, output_dir):
    model = SimpleCNN(num_classes=NUM_CLASSES_CIFAR).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_CNN, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    training_stats = []
    with mlflow.start_run(run_name="cnn_training"):
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
        mlflow.pytorch.log_model(model, "cnn_model")

        torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
    return model, training_stats

def train_bert(train_loader, test_loader, output_dir):
    logger.info("Starting BERT training...")
    try:
        # Загрузка данных
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
        
        train_input_ids = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'))
        train_attention_masks = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'))
        train_labels = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'))
        test_input_ids = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'))
        test_attention_masks = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'))
        test_labels = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'))
        logger.info("IMDB data loaded successfully")

        # Проверка формата данных
        logger.info(f"Train input_ids shape: {train_input_ids.shape}, dtype: {train_input_ids.dtype}")
        logger.info(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")

        # Создание датасетов
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        logger.info(f"Created DataLoaders: train batches={len(train_loader)}, test batches={len(test_loader)}")

        # Инициализация модели
        logger.info("Initializing BERT model...")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_BERT, weight_decay=0.01)

        # Обучение
        training_stats = []
        with mlflow.start_run(run_name="bert_training"):
            for epoch in range(EPOCHS_BERT):
                model.train()
                total_train_loss = 0
                for batch in tqdm(train_loader, desc=f"BERT Epoch {epoch+1}"):
                    b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    model.zero_grad()
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs.loss
                    total_train_loss += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

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
            os.makedirs(OUTPUT_DIR_BERT, exist_ok=True)
            metrics_path = os.path.join(OUTPUT_DIR_BERT, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(training_stats, f)
            mlflow.log_artifact(metrics_path)
            mlflow.pytorch.log_model(model, "bert_model")
            logger.info(f"Saving BERT model to {OUTPUT_DIR_BERT}")
            model.save_pretrained(OUTPUT_DIR_BERT)
            logger.info(f"BERT training completed, metrics saved to {metrics_path}")

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

    training_stats = []
    with mlflow.start_run(run_name=f"{rnn_type.lower()}_training"):
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

    results = {}
    with mlflow.start_run(run_name="boosting_training"):
        for name, (model, param_grid) in models.items():
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            results[name] = {
                'model': best_model,
                'accuracy': accuracy_score(y_test, y_pred),
                'val_scores': grid_search.cv_results_['mean_test_score'],
                'epochs': np.arange(1, len(grid_search.cv_results_['mean_test_score']) + 1)
            }

            mlflow.log_metric(f"{name}_accuracy", results[name]['accuracy'])
            for i, score in enumerate(results[name]['val_scores']):
                mlflow.log_metric(f"{name}_val_score", score, step=i)

            os.makedirs(output_dir, exist_ok=True)
            if name == 'XGBoost':
                best_model.save_model(os.path.join(output_dir, 'xgboost_model.json'))
                mlflow.log_artifact(os.path.join(output_dir, 'xgboost_model.json'))
            elif name == 'LightGBM':
                with open(os.path.join(output_dir, 'lightgbm_model.pkl'), 'wb') as f:
                    pickle.dump(best_model, f)
                mlflow.log_artifact(os.path.join(output_dir, 'lightgbm_model.pkl'))
            else:
                best_model.save_model(os.path.join(output_dir, 'catboost_model.cbm'))
                mlflow.log_artifact(os.path.join(output_dir, 'catboost_model.cbm'))

        with open(os.path.join(output_dir, 'boosting_metrics.pkl'), 'wb') as f:
            pickle.dump(results, f)
        mlflow.log_artifact(os.path.join(output_dir, 'boosting_metrics.pkl'))

    return results
<<<<<<< HEAD
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
import catboost as cb
import pickle
from tqdm import tqdm
from mentorex2.mentorex2.config import (
    NUM_CLASSES_CIFAR, EPOCHS_VIT, EPOCHS_CNN, LEARNING_RATE_VIT, LEARNING_RATE_CNN, WEIGHT_DECAY, LABEL_SMOOTHING,
    EPOCHS_BERT, LEARNING_RATE_BERT, EPOCHS_RNN, LEARNING_RATE_RNN, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
    DROPOUT, XGBOOST_PARAM_GRID, LIGHTGBM_PARAM_GRID, CATBOOST_PARAM_GRID, OUTPUT_DIR_VIT, OUTPUT_DIR_CNN, OUTPUT_DIR_BERT,
    OUTPUT_DIR_RNN, OUTPUT_DIR_BOOSTING, BATCH_SIZE_RNN
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    train_losses = []
    test_accuracies = []

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
        train_losses.append(running_loss / len(train_loader))

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
        test_accuracies.append(100 * correct / total)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    return model, train_losses, test_accuracies


def train_cnn(train_loader, test_loader, output_dir):
    model = SimpleCNN(num_classes=NUM_CLASSES_CIFAR).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_CNN, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses = []
    test_accuracies = []

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
        train_losses.append(running_loss / len(train_loader))

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
        test_accuracies.append(100 * correct / total)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
    return model, train_losses, test_accuracies


def train_bert(train_loader, test_loader, output_dir):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_BERT, weight_decay=0.01)

    training_stats = []
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
            total_eval_accuracy += np.sum(np.argmax(logits.detach().cpu().numpy(), axis=1) == b_labels.cpu().numpy()) / b_labels.size(0)

        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': total_eval_loss / len(test_loader),
            'Valid. Accur.': total_eval_accuracy / len(test_loader)
        })

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    return model, training_stats


def train_rnn(train_data, test_data, output_dir, rnn_type='LSTM'):
    train_loader = torch.utils.data.DataLoader(train_data, sampler=torch.utils.data.RandomSampler(train_data), batch_size=BATCH_SIZE_RNN)
    test_loader = torch.utils.data.DataLoader(test_data, sampler=torch.utils.data.SequentialSampler(test_data), batch_size=BATCH_SIZE_RNN)

    model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, rnn_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_RNN, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    training_stats = []
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

        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': total_eval_loss / len(test_loader),
            'Valid. Accur.': total_eval_acc / len(test_loader)
        })

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f'{rnn_type.lower()}_model.pth'))
    return model, training_stats


def train_boosting(X_train, y_train, X_test, y_test, output_dir):
    models = {
        'XGBoost': (xgb.XGBClassifier(eval_metric='logloss'), XGBOOST_PARAM_GRID),
        'LightGBM': (lgb.LGBMClassifier(), LIGHTGBM_PARAM_GRID),
        'CatBoost': (cb.CatBoostClassifier(verbose=0, train_dir=os.path.join(output_dir, 'catboost_info')), CATBOOST_PARAM_GRID)
    }

    results = {}
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
        os.makedirs(output_dir, exist_ok=True)
        if name == 'XGBoost':
            best_model.save_model(os.path.join(output_dir, 'xgboost_model.json'))
        elif name == 'LightGBM':
            with open(os.path.join(output_dir, 'lightgbm_model.pkl'), 'wb') as f:
                pickle.dump(best_model, f)
        else:
            best_model.save_model(os.path.join(output_dir, 'catboost_model.cbm'))

    return results
=======
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
import catboost as cb
import pickle
from tqdm import tqdm
from mentorex2.mentorex2.config import (
    NUM_CLASSES_CIFAR, EPOCHS_VIT, EPOCHS_CNN, LEARNING_RATE_VIT, LEARNING_RATE_CNN, WEIGHT_DECAY, LABEL_SMOOTHING,
    EPOCHS_BERT, LEARNING_RATE_BERT, EPOCHS_RNN, LEARNING_RATE_RNN, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS,
    DROPOUT, XGBOOST_PARAM_GRID, LIGHTGBM_PARAM_GRID, CATBOOST_PARAM_GRID, OUTPUT_DIR_VIT, OUTPUT_DIR_CNN, OUTPUT_DIR_BERT,
    OUTPUT_DIR_RNN, OUTPUT_DIR_BOOSTING, BATCH_SIZE_RNN
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    train_losses = []
    test_accuracies = []

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
        train_losses.append(running_loss / len(train_loader))

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
        test_accuracies.append(100 * correct / total)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    return model, train_losses, test_accuracies


def train_cnn(train_loader, test_loader, output_dir):
    model = SimpleCNN(num_classes=NUM_CLASSES_CIFAR).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE_CNN, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses = []
    test_accuracies = []

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
        train_losses.append(running_loss / len(train_loader))

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
        test_accuracies.append(100 * correct / total)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'cnn_model.pth'))
    return model, train_losses, test_accuracies


def train_bert(train_loader, test_loader, output_dir):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_BERT, weight_decay=0.01)

    training_stats = []
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
            total_eval_accuracy += np.sum(np.argmax(logits.detach().cpu().numpy(), axis=1) == b_labels.cpu().numpy()) / b_labels.size(0)

        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': total_eval_loss / len(test_loader),
            'Valid. Accur.': total_eval_accuracy / len(test_loader)
        })

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    return model, training_stats


def train_rnn(train_data, test_data, output_dir, rnn_type='LSTM'):
    train_loader = torch.utils.data.DataLoader(train_data, sampler=torch.utils.data.RandomSampler(train_data), batch_size=BATCH_SIZE_RNN)
    test_loader = torch.utils.data.DataLoader(test_data, sampler=torch.utils.data.SequentialSampler(test_data), batch_size=BATCH_SIZE_RNN)

    model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, rnn_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_RNN, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    training_stats = []
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

        training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': total_eval_loss / len(test_loader),
            'Valid. Accur.': total_eval_acc / len(test_loader)
        })

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, f'{rnn_type.lower()}_model.pth'))
    return model, training_stats


def train_boosting(X_train, y_train, X_test, y_test, output_dir):
    models = {
        'XGBoost': (xgb.XGBClassifier(eval_metric='logloss'), XGBOOST_PARAM_GRID),
        'LightGBM': (lgb.LGBMClassifier(), LIGHTGBM_PARAM_GRID),
        'CatBoost': (cb.CatBoostClassifier(verbose=0, train_dir=os.path.join(output_dir, 'catboost_info')), CATBOOST_PARAM_GRID)
    }

    results = {}
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
        os.makedirs(output_dir, exist_ok=True)
        if name == 'XGBoost':
            best_model.save_model(os.path.join(output_dir, 'xgboost_model.json'))
        elif name == 'LightGBM':
            with open(os.path.join(output_dir, 'lightgbm_model.pkl'), 'wb') as f:
                pickle.dump(best_model, f)
        else:
            best_model.save_model(os.path.join(output_dir, 'catboost_model.cbm'))

    return results
>>>>>>> bfff80e (Adding files)

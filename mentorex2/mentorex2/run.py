#!/usr/bin/env python
# coding: utf-8
"""
run.py - Script to run training, prediction, and visualization for the mentorex2 project.
"""

from mentorex2.mentorex2.config import PROCESSED_DIR, OUTPUT_DIR_VIT, OUTPUT_DIR_CNN, OUTPUT_DIR_BERT, OUTPUT_DIR_RNN, OUTPUT_DIR_BOOSTING, BATCH_SIZE_BERT, BATCH_SIZE_RNN
from mentorex2.mentorex2.plots import plot_training_metrics, plot_cnn_filters, plot_model_comparison
from mentorex2.mentorex2.modeling.predict import predict_cifar10_vit, predict_cifar10_cnn, predict_imdb_bert, predict_imdb_rnn, predict_imdb_boosting
from mentorex2.mentorex2.modeling.train import train_vit, train_cnn, train_bert, train_rnn, train_boosting
from mentorex2.mentorex2.features import load_cifar10_data, load_imdb_data
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_metrics_if_exists(model_dir, filename):
    """Load metrics from file if it exists."""
    metrics_path = os.path.join(model_dir, filename)
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            return pickle.load(f)
    return None


def load_imdb_data():
    """Load preprocessed IMDB data."""
    train_input_ids = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'))
    train_attention_masks = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'))
    train_labels_bert = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'))
    test_input_ids = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'))
    test_attention_masks = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'))
    test_labels_bert = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'))

    train_padded = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_padded_rnn.pt'))
    train_lengths = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_lengths_rnn.pt'))
    train_labels_rnn = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_rnn.pt'))
    test_padded = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_padded_rnn.pt'))
    test_lengths = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_lengths_rnn.pt'))
    test_labels_rnn = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_rnn.pt'))

    with open(os.path.join(PROCESSED_DIR, 'imdb_train_tfidf.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, 'imdb_test_tfidf.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    y_train = np.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_boosting.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_boosting.npy'))

    train_dataset_bert = TensorDataset(train_input_ids, train_attention_masks, train_labels_bert)
    test_dataset_bert = TensorDataset(test_input_ids, test_attention_masks, test_labels_bert)
    train_loader_bert = DataLoader(train_dataset_bert, batch_size=BATCH_SIZE_BERT, shuffle=True)
    test_loader_bert = DataLoader(test_dataset_bert, batch_size=BATCH_SIZE_BERT, shuffle=False)

    train_dataset_rnn = TensorDataset(train_padded, train_lengths, train_labels_rnn)
    test_dataset_rnn = TensorDataset(test_padded, test_lengths, test_labels_rnn)

    print(f"train_padded shape: {train_padded.shape}")
    print(f"train_lengths shape: {train_lengths.shape}")
    print(f"train_labels_rnn shape: {train_labels_rnn.shape}")
    print(f"Max index in train_padded: {train_padded.max().item()}")
    print(f"Min index in train_padded: {train_padded.min().item()}")
    print(f"Max length in train_lengths: {train_lengths.max().item()}")
    print(f"Min length in train_lengths: {train_lengths.min().item()}")

    return (train_loader_bert, test_loader_bert), (train_dataset_rnn, test_dataset_rnn), (X_train, X_test, y_train, y_test)


def main():
    # Load data
    print("Loading CIFAR-10 and IMDB data...")
    (train_loader_vit, test_loader_vit), (train_loader_cnn, test_loader_cnn) = load_cifar10_data()
    (train_loader_bert, test_loader_bert), (train_data_rnn, test_data_rnn), (X_train, X_test, y_train, y_test) = load_imdb_data()

    # Initialize results for comparison
    results = {'Model': [], 'Accuracy': []}

    # Train or load ViT
    vit_model_dir = os.path.join(MODELS_DIR, 'vit')
    if os.path.exists(os.path.join(vit_model_dir, 'pytorch_model.bin')) or os.path.exists(os.path.join(vit_model_dir, 'model.safetensors')):
        print("ViT model already exists, skipping training...")
        vit_metrics = load_metrics_if_exists(vit_model_dir, 'metrics.pkl')
        vit_losses = vit_metrics['train_losses'] if vit_metrics else []
        vit_acc = vit_metrics['test_accuracies'] if vit_metrics else []
    else:
        print("Training ViT...")
        vit_model, vit_losses, vit_acc = train_vit(train_loader_vit, test_loader_vit, vit_model_dir)
        with open(os.path.join(vit_model_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump({'train_losses': vit_losses, 'test_accuracies': vit_acc}, f)
    if vit_acc:
        results['Model'].append('ViT')
        results['Accuracy'].append(vit_acc[-1])

    # Train or load CNN
    cnn_model_dir = os.path.join(MODELS_DIR, 'cnn')
    if os.path.exists(os.path.join(cnn_model_dir, 'cnn_model.pth')):
        print("CNN model already exists, skipping training...")
        cnn_metrics = load_metrics_if_exists(cnn_model_dir, 'metrics.pkl')
        cnn_losses = cnn_metrics['train_losses'] if cnn_metrics else []
        cnn_acc = cnn_metrics['test_accuracies'] if cnn_metrics else []
    else:
        print("Training CNN...")
        cnn_model, cnn_losses, cnn_acc = train_cnn(train_loader_cnn, test_loader_cnn, cnn_model_dir)
        with open(os.path.join(cnn_model_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump({'train_losses': cnn_losses, 'test_accuracies': cnn_acc}, f)
    if cnn_acc:
        results['Model'].append('CNN')
        results['Accuracy'].append(cnn_acc[-1])

    # Train or load BERT
    bert_model_dir = os.path.join(MODELS_DIR, 'bert')
    if os.path.exists(os.path.join(bert_model_dir, 'pytorch_model.bin')) or os.path.exists(os.path.join(bert_model_dir, 'model.safetensors')):
        print("BERT model already exists, skipping training...")
        bert_stats = load_metrics_if_exists(bert_model_dir, 'metrics.pkl')
    else:
        print("Training BERT...")
        bert_model, bert_stats = train_bert(train_loader_bert, test_loader_bert, bert_model_dir)
        with open(os.path.join(bert_model_dir, 'metrics.pkl'), 'wb') as f:
            pickle.dump(bert_stats, f)
    if bert_stats:
        results['Model'].append('BERT')
        results['Accuracy'].append(bert_stats[-1]['Valid. Accur.'])

    # Train or load LSTM
    rnn_model_dir = os.path.join(MODELS_DIR, 'rnn')
    if os.path.exists(os.path.join(rnn_model_dir, 'lstm_model.pth')):
        print("LSTM model already exists, skipping training...")
        lstm_stats = load_metrics_if_exists(rnn_model_dir, 'lstm_metrics.pkl')
    else:
        print("Training LSTM...")
        lstm_model, lstm_stats = train_rnn(train_data_rnn, test_data_rnn, rnn_model_dir, 'LSTM')
        with open(os.path.join(rnn_model_dir, 'lstm_metrics.pkl'), 'wb') as f:
            pickle.dump(lstm_stats, f)
    if lstm_stats:
        results['Model'].append('LSTM')
        results['Accuracy'].append(lstm_stats[-1]['Valid. Accur.'])

    # Train or load GRU
    if os.path.exists(os.path.join(rnn_model_dir, 'gru_model.pth')):
        print("GRU model already exists, skipping training...")
        gru_stats = load_metrics_if_exists(rnn_model_dir, 'gru_metrics.pkl')
    else:
        print("Training GRU...")
        gru_model, gru_stats = train_rnn(train_data_rnn, test_data_rnn, rnn_model_dir, 'GRU')
        with open(os.path.join(rnn_model_dir, 'gru_metrics.pkl'), 'wb') as f:
            pickle.dump(gru_stats, f)
    if gru_stats:
        results['Model'].append('GRU')
        results['Accuracy'].append(gru_stats[-1]['Valid. Accur.'])

    # Train boosting models
    boosting_model_dir = os.path.join(MODELS_DIR, 'boosting')
    if (os.path.exists(os.path.join(boosting_model_dir, 'xgboost_model.json'))
        and os.path.exists(os.path.join(boosting_model_dir, 'lightgbm_model.pkl'))
            and os.path.exists(os.path.join(boosting_model_dir, 'catboost_model.cbm'))):
        print("Boosting models already exist, skipping training...")
        boosting_results = load_metrics_if_exists(boosting_model_dir, 'boosting_metrics.pkl')
    else:
        print("Training boosting models...")
        try:
            boosting_results = train_boosting(X_train, y_train, X_test, y_test, boosting_model_dir)
            with open(os.path.join(boosting_model_dir, 'boosting_metrics.pkl'), 'wb') as f:
                pickle.dump(boosting_results, f)
        except KeyboardInterrupt:
            print("Training interrupted. Saving partial boosting results if available...")
            boosting_results = load_metrics_if_exists(boosting_model_dir, 'boosting_metrics.pkl') or {}
    for name in ['XGBoost', 'LightGBM', 'CatBoost']:
        if boosting_results and name in boosting_results:
            results['Model'].append(name)
            results['Accuracy'].append(boosting_results[name]['accuracy'])

# Пасхалка
    # Plot results
    print("Generating plots...")
    if vit_losses:
        plot_training_metrics(vit_losses, 'ViT', os.path.join(FIGURES_DIR, 'vit_results.png'))
    if cnn_losses:
        plot_training_metrics(cnn_losses, 'CNN', os.path.join(FIGURES_DIR, 'cnn_results.png'))
        if os.path.exists(os.path.join(cnn_model_dir, 'cnn_model.pth')):
            plot_cnn_filters(os.path.join(cnn_model_dir, 'cnn_model.pth'), test_loader_cnn, os.path.join(FIGURES_DIR, 'cnn_filters.png'))
    if bert_stats:
        plot_training_metrics(bert_stats, 'BERT', os.path.join(FIGURES_DIR, 'bert_results.png'))
    if lstm_stats:
        plot_training_metrics(lstm_stats, 'LSTM', os.path.join(FIGURES_DIR, 'lstm_results.png'))
    if gru_stats:
        plot_training_metrics(gru_stats, 'GRU', os.path.join(FIGURES_DIR, 'gru_results.png'))
    if boosting_results:
        for name in ['XGBoost', 'LightGBM', 'CatBoost']:
            if name in boosting_results:
                plot_training_metrics(boosting_results[name]['val_scores'], name, os.path.join(FIGURES_DIR, f'{name.lower()}_results.png'))

    # Compare models
    if results['Model']:
        plot_model_comparison(results, os.path.join(FIGURES_DIR, 'model_comparison.png'))

    # Example predictions
    print("Running example predictions...")
    image_path = r'C:\Users\Delta-Game\mentorex2\mentorex2\data\sample_cifar10_image.jpg'
    if os.path.exists(image_path):
        print(f"ViT Prediction: {predict_cifar10_vit(image_path)}")
        print(f"CNN Prediction: {predict_cifar10_cnn(image_path)}")
    else:
        print(f"Image path {image_path} not found. Skipping CIFAR-10 predictions.")

    text = "This movie was fantastic!"
    if os.path.exists(os.path.join(bert_model_dir, 'pytorch_model.bin')) or os.path.exists(os.path.join(bert_model_dir, 'model.safetensors')):
        print(f"BERT Prediction: {predict_imdb_bert(text)}")
    if os.path.exists(os.path.join(rnn_model_dir, 'lstm_model.pth')):
        print(f"LSTM Prediction: {predict_imdb_rnn(text, rnn_type='LSTM')}")
    if os.path.exists(os.path.join(rnn_model_dir, 'gru_model.pth')):
        print(f"GRU Prediction: {predict_imdb_rnn(text, rnn_type='GRU')}")
    if boosting_results and 'XGBoost' in boosting_results:
        print(f"XGBoost Prediction: {predict_imdb_boosting(text, 'XGBoost')}")
    if boosting_results and 'LightGBM' in boosting_results:
        print(f"LightGBM Prediction: {predict_imdb_boosting(text, 'LightGBM')}")
    if boosting_results and 'CatBoost' in boosting_results:
        print(f"CatBoost Prediction: {predict_imdb_boosting(text, 'CatBoost')}")


if __name__ == "__main__":
    main()
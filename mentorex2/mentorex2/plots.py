#!/usr/bin/env python
# coding: utf-8
"""
plots.py - Generate visualizations for model performance in the mentorex2 project.
"""

from mentorex2.mentorex2.modeling.train import SimpleCNN
from mentorex2.mentorex2.config import OUTPUT_DIR_VIT, OUTPUT_DIR_CNN, OUTPUT_DIR_BERT, OUTPUT_DIR_RNN, OUTPUT_DIR_BOOSTING
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import pickle

# Настройка пути к модулю mentorex2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Определяем FIGURES_DIR вручную
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'reports', 'figures')


def load_metrics_if_exists(model_dir, filename):
    """Load metrics from file if it exists."""
    metrics_path = os.path.join(model_dir, filename)
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            return pickle.load(f)
    return None


def plot_training_metrics(data, model_name, save_path):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))

    # Обработка метрик для ViT, CNN
    if isinstance(data, dict) and 'train_losses' in data and 'test_accuracies' in data:
        epochs = range(1, len(data['train_losses']) + 1)
        plt.subplot(1, 2, 1)
        plt.plot(epochs, data['train_losses'], label='Train Loss')
        plt.title(f'{model_name} Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, data['test_accuracies'], label='Val Acc')
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

    # Обработка метрик для BERT, LSTM, GRU
    elif isinstance(data, list) and 'Training Loss' in data[0]:
        plt.subplot(1, 2, 1)
        plt.plot([x['epoch'] for x in data], [x['Training Loss'] for x in data], label='Train Loss')
        plt.plot([x['epoch'] for x in data], [x['Valid. Loss'] for x in data], label='Val Loss')
        plt.title(f'{model_name} Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot([x['epoch'] for x in data], [x['Valid. Accur.'] for x in data], label='Val Acc')
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    # Обработка метрик для Boosting
    elif isinstance(data, dict) and 'val_scores' in data:
        epochs = data['epochs']
        plt.subplot(1, 2, 1)
        plt.plot(epochs, data['val_scores'], label='Val Score')
        plt.title(f'{model_name} Validation Scores')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot([epochs[-1]], [data['accuracy']], 'o', label='Final Acc')
        plt.title(f'{model_name} Final Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_cnn_filters(model_path, test_loader, save_path):
    """Visualize CNN filters and feature maps."""
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Визуализация фильтров
    filters = model.conv1.weight.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    for i in range(min(16, filters.shape[0])):
        plt.subplot(4, 4, i + 1)
        filter_rgb = filters[i].transpose(1, 2, 0)
        filter_rgb = (filter_rgb - filter_rgb.min()) / (filter_rgb.max() - filter_rgb.min())
        plt.imshow(filter_rgb)
        plt.axis('off')
    plt.suptitle('Conv1 Filters')
    plt.savefig(save_path.replace('.png', '_filters.png'))
    plt.close()

    # Визуализация feature maps
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images[:1].to(model.conv1.weight.device)
        conv1_output = model.conv1(images).cpu().numpy()
        plt.figure(figsize=(10, 10))
        for i in range(min(16, conv1_output.shape[1])):
            plt.subplot(4, 4, i + 1)
            plt.imshow(conv1_output[0, i], cmap='gray')
            plt.axis('off')
        plt.suptitle('Conv1 Feature Maps')
        plt.savefig(save_path.replace('.png', '_feature_maps.png'))
        plt.close()


def plot_model_comparison(results, save_path):
    """Plot comparison of model accuracies."""
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 4))
    sns.barplot(x='Model', y='Accuracy', data=results_df)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.savefig(save_path)
    plt.close()


def main():
    """Generate plots for all models."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load metrics
    vit_metrics = load_metrics_if_exists(OUTPUT_DIR_VIT, 'metrics.pkl')
    cnn_metrics = load_metrics_if_exists(OUTPUT_DIR_CNN, 'metrics.pkl')
    bert_metrics = load_metrics_if_exists(OUTPUT_DIR_BERT, 'metrics.pkl')
    lstm_metrics = load_metrics_if_exists(OUTPUT_DIR_RNN, 'lstm_metrics.pkl')
    gru_metrics = load_metrics_if_exists(OUTPUT_DIR_RNN, 'gru_metrics.pkl')
    boosting_metrics = load_metrics_if_exists(OUTPUT_DIR_BOOSTING, 'boosting_metrics.pkl')

    # Load CIFAR-10 test loader for CNN visualization
    from mentorex2.mentorex2.features import load_cifar10_data
    _, (test_loader_cnn, _) = load_cifar10_data()

    # Plot metrics
    if vit_metrics:
        plot_training_metrics(vit_metrics, 'ViT', os.path.join(FIGURES_DIR, 'vit_results.png'))
    if cnn_metrics:
        plot_training_metrics(cnn_metrics, 'CNN', os.path.join(FIGURES_DIR, 'cnn_results.png'))
        if os.path.exists(os.path.join(OUTPUT_DIR_CNN, 'cnn_model.pth')):
            plot_cnn_filters(os.path.join(OUTPUT_DIR_CNN, 'cnn_model.pth'), test_loader_cnn, os.path.join(FIGURES_DIR, 'cnn_results.png'))
    if bert_metrics:
        plot_training_metrics(bert_metrics, 'BERT', os.path.join(FIGURES_DIR, 'bert_results.png'))
    if lstm_metrics:
        plot_training_metrics(lstm_metrics, 'LSTM', os.path.join(FIGURES_DIR, 'lstm_results.png'))
    if gru_metrics:
        plot_training_metrics(gru_metrics, 'GRU', os.path.join(FIGURES_DIR, 'gru_results.png'))
    if boosting_metrics:
        for name in ['XGBoost', 'LightGBM', 'CatBoost']:
            if name in boosting_metrics:
                plot_training_metrics(boosting_metrics[name], name, os.path.join(FIGURES_DIR, f'{name.lower()}_results.png'))

    # Plot model comparison
    results = {'Model': [], 'Accuracy': []}
    if vit_metrics and 'test_accuracies' in vit_metrics:
        results['Model'].append('ViT')
        results['Accuracy'].append(vit_metrics['test_accuracies'][-1])
    if cnn_metrics and 'test_accuracies' in cnn_metrics:
        results['Model'].append('CNN')
        results['Accuracy'].append(cnn_metrics['test_accuracies'][-1])
    if bert_metrics:
        results['Model'].append('BERT')
        results['Accuracy'].append(bert_metrics[-1]['Valid. Accur.'])
    if lstm_metrics:
        results['Model'].append('LSTM')
        results['Accuracy'].append(lstm_metrics[-1]['Valid. Accur.'])
    if gru_metrics:
        results['Model'].append('GRU')
        results['Accuracy'].append(gru_metrics[-1]['Valid. Accur.'])
    if boosting_metrics:
        for name in ['XGBoost', 'LightGBM', 'CatBoost']:
            if name in boosting_metrics:
                results['Model'].append(name)
                results['Accuracy'].append(boosting_metrics[name]['accuracy'])

    if results['Model']:
        plot_model_comparison(results, os.path.join(FIGURES_DIR, 'model_comparison.png'))


if __name__ == "__main__":
    main()

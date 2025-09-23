#!/usr/bin/env python
# coding: utf-8
"""
config.py - Store useful variables and configuration for the mentorex2 project.
"""

import os

# Base paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'mentorex2', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'mentorex2', 'models')
OUTPUT_DIR_VIT = os.path.join(MODEL_DIR, 'vit')
OUTPUT_DIR_CNN = os.path.join(MODEL_DIR, 'cnn')
OUTPUT_DIR_BERT = os.path.join(MODEL_DIR, 'bert')
OUTPUT_DIR_RNN = os.path.join(MODEL_DIR, 'rnn')
OUTPUT_DIR_BOOSTING = os.path.join(MODEL_DIR, 'boosting')
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'mentorex2', 'logs')

# CIFAR-10 parameters
NUM_CLASSES_CIFAR = 10
BATCH_SIZE_CIFAR = 64
EPOCHS_VIT = 5
EPOCHS_CNN = 50
LEARNING_RATE_VIT = 1e-4
LEARNING_RATE_CNN = 1e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1

# IMDB parameters
BATCH_SIZE_BERT = 16
BATCH_SIZE_RNN = 64
EPOCHS_BERT = 3
EPOCHS_RNN = 10
LEARNING_RATE_BERT = 2e-5
LEARNING_RATE_RNN = 0.0005
VOCAB_SIZE = 20000
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
MAX_LENGTH_BERT = 512
MAX_LENGTH_RNN = 256
TFIDF_MAX_FEATURES = 5000
BATCH_SIZE_RNN = 64

# Boosting parameters
TFIDF_MAX_FEATURES = 5000
XGBOOST_PARAM_GRID = {
    'max_depth': [3],
    'n_estimators': [50]
}
LIGHTGBM_PARAM_GRID = {
    'max_depth': [3],
    'n_estimators': [50]
}
CATBOOST_PARAM_GRID = {
    'depth': [3],
    'iterations': [50]
}
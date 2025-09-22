#!/usr/bin/env python
# coding: utf-8
"""
features.py - Code to load preprocessed data and create DataLoaders for modeling.
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
import pickle
from mentorex2.mentorex2.config import PROCESSED_DIR, BATCH_SIZE_CIFAR, BATCH_SIZE_BERT, BATCH_SIZE_RNN, VOCAB_SIZE, MAX_LENGTH_RNN


def load_cifar10_data():
    """Load preprocessed CIFAR-10 data for ViT and CNN."""
    print("Loading CIFAR-10 data from", PROCESSED_DIR)
    # Load ViT data
    train_images_vit = np.load(os.path.join(PROCESSED_DIR, 'cifar10_train_images_vit.npy'), mmap_mode='r')
    train_labels_vit = np.load(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_vit.npy'), mmap_mode='r')
    test_images_vit = np.load(os.path.join(PROCESSED_DIR, 'cifar10_test_images_vit.npy'), mmap_mode='r')
    test_labels_vit = np.load(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_vit.npy'), mmap_mode='r')

    train_dataset_vit = TensorDataset(
        torch.tensor(train_images_vit, dtype=torch.float32),
        torch.tensor(train_labels_vit, dtype=torch.long)
    )
    test_dataset_vit = TensorDataset(
        torch.tensor(test_images_vit, dtype=torch.float32),
        torch.tensor(test_labels_vit, dtype=torch.long)
    )

    train_loader_vit = DataLoader(train_dataset_vit, batch_size=BATCH_SIZE_CIFAR, shuffle=True)
    test_loader_vit = DataLoader(test_dataset_vit, batch_size=BATCH_SIZE_CIFAR, shuffle=False)

    # Load CNN data
    train_images_cnn = np.load(os.path.join(PROCESSED_DIR, 'cifar10_train_images_cnn.npy'), mmap_mode='r')
    train_labels_cnn = np.load(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_cnn.npy'), mmap_mode='r')
    test_images_cnn = np.load(os.path.join(PROCESSED_DIR, 'cifar10_test_images_cnn.npy'), mmap_mode='r')
    test_labels_cnn = np.load(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_cnn.npy'), mmap_mode='r')

    train_dataset_cnn = TensorDataset(
        torch.tensor(train_images_cnn, dtype=torch.float32),
        torch.tensor(train_labels_cnn, dtype=torch.long)
    )
    test_dataset_cnn = TensorDataset(
        torch.tensor(test_images_cnn, dtype=torch.float32),
        torch.tensor(test_labels_cnn, dtype=torch.long)
    )

    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=BATCH_SIZE_CIFAR, shuffle=True)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=BATCH_SIZE_CIFAR, shuffle=False)

    return (train_loader_vit, test_loader_vit), (train_loader_cnn, test_loader_cnn)


def load_imdb_data():
    """Load preprocessed IMDB data for BERT, RNN, and boosting."""
    print("Loading IMDB data from", PROCESSED_DIR)
    # Load BERT data
    train_input_ids = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'))
    train_attention_masks = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'))
    train_labels_bert = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'))
    test_input_ids = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'))
    test_attention_masks = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'))
    test_labels_bert = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'))

    train_data_bert = TensorDataset(train_input_ids, train_attention_masks, train_labels_bert)
    test_data_bert = TensorDataset(test_input_ids, test_attention_masks, test_labels_bert)

    train_loader_bert = DataLoader(train_data_bert, sampler=RandomSampler(train_data_bert), batch_size=BATCH_SIZE_BERT)
    test_loader_bert = DataLoader(test_data_bert, sampler=SequentialSampler(test_data_bert), batch_size=BATCH_SIZE_BERT)

    # Load RNN data
    train_padded = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_padded_rnn.pt'))
    train_lengths = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_lengths_rnn.pt'))
    train_labels_rnn = torch.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_rnn.pt'))
    test_padded = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_padded_rnn.pt'))
    test_lengths = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_lengths_rnn.pt'))
    test_labels_rnn = torch.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_rnn.pt'))

    # Debug: Check RNN data integrity
    print("train_padded shape:", train_padded.shape)
    print("train_lengths shape:", train_lengths.shape)
    print("train_labels_rnn shape:", train_labels_rnn.shape)
    print("Max index in train_padded:", train_padded.max().item())
    print("Min index in train_padded:", train_padded.min().item())
    print("Max length in train_lengths:", train_lengths.max().item())
    print("Min length in train_lengths:", train_lengths.min().item())

    # Validate indices
    if train_padded.max().item() >= VOCAB_SIZE:
        raise ValueError(f"train_padded contains indices >= VOCAB_SIZE ({VOCAB_SIZE})")
    if train_padded.min().item() < 0:
        raise ValueError("train_padded contains negative indices")
    if train_lengths.min().item() <= 0:
        raise ValueError("train_lengths contains zero or negative lengths")

    train_data_rnn = TensorDataset(train_padded, train_lengths, train_labels_rnn)
    test_data_rnn = TensorDataset(test_padded, test_lengths, test_labels_rnn)

    # Use SequentialSampler to avoid shuffling, as data will be sorted in train_rnn
    train_loader_rnn = DataLoader(train_data_rnn, sampler=SequentialSampler(train_data_rnn), batch_size=BATCH_SIZE_RNN)
    test_loader_rnn = DataLoader(test_data_rnn, sampler=SequentialSampler(test_data_rnn), batch_size=BATCH_SIZE_RNN)

    # Load boosting data
    with open(os.path.join(PROCESSED_DIR, 'imdb_train_tfidf.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, 'imdb_test_tfidf.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    train_labels_boosting = np.load(os.path.join(PROCESSED_DIR, 'imdb_train_labels_boosting.npy'))
    test_labels_boosting = np.load(os.path.join(PROCESSED_DIR, 'imdb_test_labels_boosting.npy'))

    return (train_loader_bert, test_loader_bert), (train_data_rnn, test_data_rnn), (X_train, X_test, train_labels_boosting, test_labels_boosting)

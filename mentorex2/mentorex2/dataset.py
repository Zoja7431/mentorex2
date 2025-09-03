#!/usr/bin/env python
# coding: utf-8
"""
dataset.py - Scripts to download or generate data for the mentorex2 project.
Saves interim and processed datasets for CIFAR-10 and IMDB.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk

# Убедимся, что нужные NLTK-данные загружены
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Определение путей с учетом тройной вложенности и правильного расположения папки data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'mentorex2', 'data')  # Папка data в mentorex2/mentorex2/data
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Создание директорий
for directory in [INTERIM_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)

def load_cifar10_data(data_dir=RAW_DIR):
    """
    Загружает CIFAR-10 датасет из локальной папки.
    Сохраняет interim и processed данные.
    """
    cifar10_dir = os.path.join(data_dir, 'cifar10')
    
    if not os.path.exists(os.path.join(cifar10_dir, 'train')) or not os.path.exists(os.path.join(cifar10_dir, 'test')):
        raise FileNotFoundError(f"CIFAR-10 data not found in {cifar10_dir}. Ensure train and test folders exist.")

    # Трансформации для interim (без аугментации, только ToTensor)
    interim_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Трансформации для processed (нормализация и ресайз для ViT)
    processed_transform_vit = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])
    
    processed_transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])

    # Загрузка датасета
    train_dataset = datasets.ImageFolder(
        root=os.path.join(cifar10_dir, 'train'),
        transform=interim_transform
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(cifar10_dir, 'test'),
        transform=interim_transform
    )
# Aboba
    # Сохранение interim данных (сырые тензоры и метки) 
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    for img, label in train_dataset:
        train_images.append(img.numpy())
        train_labels.append(label)
    for img, label in test_dataset:
        test_images.append(img.numpy())
        test_labels.append(label)
    
    np.save(os.path.join(INTERIM_DIR, 'cifar10_train_images.npy'), np.array(train_images))
    np.save(os.path.join(INTERIM_DIR, 'cifar10_train_labels.npy'), np.array(train_labels))
    np.save(os.path.join(INTERIM_DIR, 'cifar10_test_images.npy'), np.array(test_images))
    np.save(os.path.join(INTERIM_DIR, 'cifar10_test_labels.npy'), np.array(test_labels))
    print(f"CIFAR-10 interim data saved in {INTERIM_DIR}")

    # Сохранение processed данных (нормализованные для ViT и CNN)
    train_dataset_vit = datasets.ImageFolder(
        root=os.path.join(cifar10_dir, 'train'),
        transform=processed_transform_vit
    )
    test_dataset_vit = datasets.ImageFolder(
        root=os.path.join(cifar10_dir, 'test'),
        transform=processed_transform_vit
    )
    train_dataset_cnn = datasets.ImageFolder(
        root=os.path.join(cifar10_dir, 'train'),
        transform=processed_transform_cnn
    )
    test_dataset_cnn = datasets.ImageFolder(
        root=os.path.join(cifar10_dir, 'test'),
        transform=processed_transform_cnn
    )

    # Сохранение processed для ViT
    train_images_vit = []
    train_labels_vit = []
    test_images_vit = []
    test_labels_vit = []
    for img, label in train_dataset_vit:
        train_images_vit.append(img.numpy())
        train_labels_vit.append(label)
    for img, label in test_dataset_vit:
        test_images_vit.append(img.numpy())
        test_labels_vit.append(label)
    
    np.save(os.path.join(PROCESSED_DIR, 'cifar10_train_images_vit.npy'), np.array(train_images_vit))
    np.save(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_vit.npy'), np.array(train_labels_vit))
    np.save(os.path.join(PROCESSED_DIR, 'cifar10_test_images_vit.npy'), np.array(test_images_vit))
    np.save(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_vit.npy'), np.array(test_labels_vit))

    # Сохранение processed для CNN
    train_images_cnn = []
    train_labels_cnn = []
    test_images_cnn = []
    test_labels_cnn = []
    for img, label in train_dataset_cnn:
        train_images_cnn.append(img.numpy())
        train_labels_cnn.append(label)
    for img, label in test_dataset_cnn:
        test_images_cnn.append(img.numpy())
        test_labels_cnn.append(label)
    
    np.save(os.path.join(PROCESSED_DIR, 'cifar10_train_images_cnn.npy'), np.array(train_images_cnn))
    np.save(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_cnn.npy'), np.array(train_labels_cnn))
    np.save(os.path.join(PROCESSED_DIR, 'cifar10_test_images_cnn.npy'), np.array(test_images_cnn))
    np.save(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_cnn.npy'), np.array(test_labels_cnn))
    print(f"CIFAR-10 processed data saved in {PROCESSED_DIR}")

def load_imdb_data(data_dir=RAW_DIR):
    """
    Загружает IMDB датасет из CSV, выполняет предобработку и сохраняет interim и processed данные.
    """
    imdb_path = os.path.join(data_dir, 'IMDB Dataset.csv')
    if not os.path.exists(imdb_path):
        raise FileNotFoundError(f"IMDB Dataset not found at {imdb_path}")

    df = pd.read_csv(imdb_path)
    
    # Предобработка текста
    stop_words = set(stopwords.words('english'))
    def clean_text(text):
        text = re.sub(r'<br />', ' ', text.lower())
        text = re.sub(r'[^a-z ]', '', text)
        tokens = word_tokenize(text)
        return [w for w in tokens if w not in stop_words]
    
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Разделение на тренировочную и тестовую выборки
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned_review'].tolist(),
        df['sentiment'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )

    # Сохранение interim данных (очищенные тексты и метки)
    interim_data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels
    }
    with open(os.path.join(INTERIM_DIR, 'imdb_interim.pkl'), 'wb') as f:
        pickle.dump(interim_data, f)
    print(f"IMDB interim data saved in {INTERIM_DIR}")

    # Подготовка processed данных
    # 1. Для BERT (токенизация)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length_bert = 512

    def tokenize_data(texts, labels, max_length):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels

    train_input_ids, train_attention_masks, train_labels_bert = tokenize_data(train_texts, train_labels, max_length_bert)
    test_input_ids, test_attention_masks, test_labels_bert = tokenize_data(test_texts, test_labels, max_length_bert)

    # Сохранение processed данных для BERT
    torch.save(train_input_ids, os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'))
    torch.save(train_attention_masks, os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'))
    torch.save(train_labels_bert, os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'))
    torch.save(test_input_ids, os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'))
    torch.save(test_attention_masks, os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'))
    torch.save(test_labels_bert, os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'))
    print(f"IMDB processed data (BERT) saved in {PROCESSED_DIR}")

    # 2. Для RNN (LSTM/GRU)
    vocab_size = 20000
    max_length_rnn = 256
    all_train_words = [word for text in train_texts for word in text]
    word_counts = Counter(all_train_words)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common(vocab_size))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1

    def text_to_sequence(text, vocab, max_length):
        seq = [vocab.get(word, vocab['<UNK>']) for word in text]
        if len(seq) > max_length:
            seq = seq[:max_length]
        return seq

    train_sequences = [torch.tensor(text_to_sequence(text, vocab, max_length_rnn)) for text in train_texts]
    test_sequences = [torch.tensor(text_to_sequence(text, vocab, max_length_rnn)) for text in test_texts]
    train_lengths = [min(len(seq), max_length_rnn) for seq in train_sequences]
    test_lengths = [min(len(seq), max_length_rnn) for seq in test_sequences]
    train_padded = torch.nn.utils.rnn.pad_sequence(train_sequences, batch_first=True, padding_value=0)
    test_padded = torch.nn.utils.rnn.pad_sequence(test_sequences, batch_first=True, padding_value=0)

    # Сохранение processed данных для RNN
    torch.save(train_padded, os.path.join(PROCESSED_DIR, 'imdb_train_padded_rnn.pt'))
    torch.save(torch.tensor(train_lengths), os.path.join(PROCESSED_DIR, 'imdb_train_lengths_rnn.pt'))
    torch.save(torch.tensor(train_labels), os.path.join(PROCESSED_DIR, 'imdb_train_labels_rnn.pt'))
    torch.save(test_padded, os.path.join(PROCESSED_DIR, 'imdb_test_padded_rnn.pt'))
    torch.save(torch.tensor(test_lengths), os.path.join(PROCESSED_DIR, 'imdb_test_lengths_rnn.pt'))
    torch.save(torch.tensor(test_labels), os.path.join(PROCESSED_DIR, 'imdb_test_labels_rnn.pt'))
    with open(os.path.join(PROCESSED_DIR, 'imdb_vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    print(f"IMDB processed data (RNN) saved in {PROCESSED_DIR}")

    # 3. Для градиентного бустинга (TF-IDF)
    train_texts_str = [' '.join(text) for text in train_texts]
    test_texts_str = [' '.join(text) for text in test_texts]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts_str)
    X_test = vectorizer.transform(test_texts_str)

    # Сохранение processed данных для бустинга
    with open(os.path.join(PROCESSED_DIR, 'imdb_train_tfidf.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(PROCESSED_DIR, 'imdb_test_tfidf.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(PROCESSED_DIR, 'imdb_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    np.save(os.path.join(PROCESSED_DIR, 'imdb_train_labels_boosting.npy'), np.array(train_labels))
    np.save(os.path.join(PROCESSED_DIR, 'imdb_test_labels_boosting.npy'), np.array(test_labels))
    print(f"IMDB processed data (Boosting) saved in {PROCESSED_DIR}")


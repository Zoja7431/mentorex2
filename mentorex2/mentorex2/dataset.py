#!/usr/bin/env python
# coding: utf-8
"""
dataset.py - Script to preprocess CIFAR-10 (ViT, CNN) and IMDB (BERT, RNN, Boosting) datasets for the mentorex2 project.
"""
import argparse
import os
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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import json  # Замени pickle на json для safety
import nltk
import logging  # Добавь logging для отладки

# Настройка logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Убедимся, что нужные NLTK-данные загружены
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Определение путей 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'mentorex2', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Создание директорий
for directory in [INTERIM_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)


def process_cifar10_vit(batch_size=1000):  # Добавь arg для batch_size
    """
    Preprocess CIFAR-10 dataset for ViT with batch processing.
    Saves interim and processed data.
    """
    cifar10_dir = os.path.join(RAW_DIR, 'cifar10')
    train_dir = os.path.join(cifar10_dir, 'train')
    test_dir = os.path.join(cifar10_dir, 'test')
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        logger.error(f"CIFAR-10 data not found in {cifar10_dir}. Ensure train and test folders exist.")
        raise FileNotFoundError(f"CIFAR-10 data not found in {cifar10_dir}. Ensure train and test folders exist.")

    # Трансформации для interim (без аугментации, только ToTensor)
    interim_transform = transforms.Compose([transforms.ToTensor()])

    # Трансформации для processed (нормализация и ресайз для ViT)
    processed_transform_vit = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])

    # Загрузка датасета
    try:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=interim_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=interim_transform)
        train_dataset_vit = datasets.ImageFolder(root=train_dir, transform=processed_transform_vit)
        test_dataset_vit = datasets.ImageFolder(root=test_dir, transform=processed_transform_vit)
    except Exception as e:
        logger.error(f"Error loading CIFAR-10 datasets: {e}")
        raise

    # DataLoaders для экономии памяти
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader_vit = DataLoader(train_dataset_vit, batch_size=batch_size, shuffle=False)
    test_loader_vit = DataLoader(test_dataset_vit, batch_size=batch_size, shuffle=False)

    # Сохранение interim данных
    try:
        with open(os.path.join(INTERIM_DIR, 'vit_cifar10_train_images.npy'), 'wb') as f_images, \
             open(os.path.join(INTERIM_DIR, 'vit_cifar10_train_labels.npy'), 'wb') as f_labels:
            for batch_images, batch_labels in train_loader:
                logger.info(f"Interim train batch shape: images {batch_images.shape}, labels {batch_labels.shape}")
                np.save(f_images, batch_images.numpy(), allow_pickle=False)
                np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
        with open(os.path.join(INTERIM_DIR, 'vit_cifar10_test_images.npy'), 'wb') as f_images, \
             open(os.path.join(INTERIM_DIR, 'vit_cifar10_test_labels.npy'), 'wb') as f_labels:
            for batch_images, batch_labels in test_loader:
                logger.info(f"Interim test batch shape: images {batch_images.shape}, labels {batch_labels.shape}")
                np.save(f_images, batch_images.numpy(), allow_pickle=False)
                np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
        logger.info(f"CIFAR-10 interim data for ViT saved in {INTERIM_DIR}")
    except Exception as e:
        logger.error(f"Error saving interim data for ViT: {e}")
        raise

    # Сохранение processed данных для ViT
    try:
        with open(os.path.join(PROCESSED_DIR, 'cifar10_train_images_vit.npy'), 'wb') as f_images, \
             open(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_vit.npy'), 'wb') as f_labels:
            for batch_images, batch_labels in train_loader_vit:
                logger.info(f"Processed train batch shape for ViT: images {batch_images.shape}, labels {batch_labels.shape}")
                np.save(f_images, batch_images.numpy(), allow_pickle=False)
                np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
        with open(os.path.join(PROCESSED_DIR, 'cifar10_test_images_vit.npy'), 'wb') as f_images, \
             open(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_vit.npy'), 'wb') as f_labels:
            for batch_images, batch_labels in test_loader_vit:
                logger.info(f"Processed test batch shape for ViT: images {batch_images.shape}, labels {batch_labels.shape}")
                np.save(f_images, batch_images.numpy(), allow_pickle=False)
                np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
        logger.info(f"CIFAR-10 processed data for ViT saved in {PROCESSED_DIR}")
    except Exception as e:
        logger.error(f"Error saving processed data for ViT: {e}")
        raise

# Аналогично для process_cifar10_cnn — добавь try/except, logging shapes, arg batch_size
def process_cifar10_cnn(batch_size=1000):
    """
    Preprocess CIFAR-10 dataset for CNN with batch processing.
    Saves processed data.
    """
    cifar10_dir = os.path.join(RAW_DIR, 'cifar10')
    train_dir = os.path.join(cifar10_dir, 'train')
    test_dir = os.path.join(cifar10_dir, 'test')
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        logger.error(f"CIFAR-10 data not found in {cifar10_dir}. Ensure train and test folders exist.")
        raise FileNotFoundError(f"CIFAR-10 data not found in {cifar10_dir}. Ensure train and test folders exist.")

    # Трансформации для processed (нормализация для CNN)
    processed_transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])

    # Загрузка датасета
    try:
        train_dataset_cnn = datasets.ImageFolder(root=train_dir, transform=processed_transform_cnn)
        test_dataset_cnn = datasets.ImageFolder(root=test_dir, transform=processed_transform_cnn)
    except Exception as e:
        logger.error(f"Error loading CIFAR-10 datasets for CNN: {e}")
        raise

    # DataLoaders для экономии памяти
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=False)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=batch_size, shuffle=False)

    # Сохранение processed данных для CNN
    try:
        with open(os.path.join(PROCESSED_DIR, 'cifar10_train_images_cnn.npy'), 'wb') as f_images, \
             open(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_cnn.npy'), 'wb') as f_labels:
            for batch_images, batch_labels in train_loader_cnn:
                logger.info(f"Processed train batch shape for CNN: images {batch_images.shape}, labels {batch_labels.shape}")
                np.save(f_images, batch_images.numpy(), allow_pickle=False)
                np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
        with open(os.path.join(PROCESSED_DIR, 'cifar10_test_images_cnn.npy'), 'wb') as f_images, \
             open(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_cnn.npy'), 'wb') as f_labels:
            for batch_images, batch_labels in test_loader_cnn:
                logger.info(f"Processed test batch shape for CNN: images {batch_images.shape}, labels {batch_labels.shape}")
                np.save(f_images, batch_images.numpy(), allow_pickle=False)
                np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
        logger.info(f"CIFAR-10 processed data for CNN saved in {PROCESSED_DIR}")
    except Exception as e:
        logger.error(f"Error saving processed data for CNN: {e}")
        raise

def process_imdb():
    """
    Preprocess IMDB dataset for BERT, RNN, and Boosting.
    Saves interim and processed data.
    """
    imdb_path = os.path.join(RAW_DIR, 'IMDB Dataset.csv')
    if not os.path.exists(imdb_path):
        logger.error(f"IMDB Dataset not found at {imdb_path}")
        raise FileNotFoundError(f"IMDB Dataset not found at {imdb_path}")

    try:
        df = pd.read_csv(imdb_path)
    except Exception as e:
        logger.error(f"Error reading IMDB CSV: {e}")
        raise

    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = re.sub(r'<br />', ' ', text.lower())
        text = re.sub(r'[^a-z ]', '', text)
        tokens = word_tokenize(text)
        return [w for w in tokens if w not in stop_words]

    df['cleaned_review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Разделение на тренировочную и тестовую выборки (80/20)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned_review'].tolist(),
        df['sentiment'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment']
    )

    # Сохранение interim данных (замени pickle на json для safety)
    interim_data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels
    }
    try:
        with open(os.path.join(INTERIM_DIR, 'imdb_interim.json'), 'w') as f:
            json.dump(interim_data, f)  # json вместо pickle — safer, no execution risk
        logger.info(f"IMDB interim data saved in {INTERIM_DIR}")
    except Exception as e:
        logger.error(f"Error saving interim data: {e}")
        raise

    # BERT preprocessing
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length_bert = 512

    def tokenize_data(texts, labels, max_length):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded_dict = tokenizer.encode_plus(
                ' '.join(text),
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
        logger.info(f"Tokenized data shapes: input_ids {input_ids.shape}, attention_masks {attention_masks.shape}, labels {labels.shape}")
        return input_ids, attention_masks, labels

    train_input_ids, train_attention_masks, train_labels_bert = tokenize_data(train_texts, train_labels, max_length_bert)
    test_input_ids, test_attention_masks, test_labels_bert = tokenize_data(test_texts, test_labels, max_length_bert)

    try:
        torch.save(train_input_ids, os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'))
        torch.save(train_attention_masks, os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'))
        torch.save(train_labels_bert, os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'))
        torch.save(test_input_ids, os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'))
        torch.save(test_attention_masks, os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'))
        torch.save(test_labels_bert, os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'))
        logger.info(f"IMDB processed data (BERT) saved in {PROCESSED_DIR}")
    except Exception as e:
        logger.error(f"Error saving BERT data: {e}")
        raise

    # RNN preprocessing
    vocab_size = 20000
    max_length_rnn = 256
    all_train_words = [word for text in train_texts for word in text]
    word_counts = Counter(all_train_words)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common(vocab_size - 2))}
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

    logger.info(f"RNN padded shapes: train_padded {train_padded.shape}, test_padded {test_padded.shape}")

    try:
        torch.save(train_padded, os.path.join(PROCESSED_DIR, 'imdb_train_padded_rnn.pt'))
        torch.save(torch.tensor(train_lengths), os.path.join(PROCESSED_DIR, 'imdb_train_lengths_rnn.pt'))
        torch.save(torch.tensor(train_labels), os.path.join(PROCESSED_DIR, 'imdb_train_labels_rnn.pt'))
        torch.save(test_padded, os.path.join(PROCESSED_DIR, 'imdb_test_padded_rnn.pt'))
        torch.save(torch.tensor(test_lengths), os.path.join(PROCESSED_DIR, 'imdb_test_lengths_rnn.pt'))
        torch.save(torch.tensor(test_labels), os.path.join(PROCESSED_DIR, 'imdb_test_labels_rnn.pt'))
        with open(os.path.join(PROCESSED_DIR, 'imdb_vocab.json'), 'w') as f:  # json вместо pickle
            json.dump(vocab, f)
        logger.info(f"IMDB processed data (RNN) saved in {PROCESSED_DIR}")
    except Exception as e:
        logger.error(f"Error saving RNN data: {e}")
        raise

    # Boosting preprocessing
    train_texts_str = [' '.join(text) for text in train_texts]
    test_texts_str = [' '.join(text) for text in test_texts]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    train_tfidf = vectorizer.fit_transform(train_texts_str)
    test_tfidf = vectorizer.transform(test_texts_str)

    logger.info(f"TF-IDF shapes: train_tfidf {train_tfidf.shape}, test_tfidf {test_tfidf.shape}")

    try:
        with open(os.path.join(PROCESSED_DIR, 'imdb_train_tfidf.pkl'), 'wb') as f:
            pickle.dump(train_tfidf, f)
        with open(os.path.join(PROCESSED_DIR, 'imdb_test_tfidf.pkl'), 'wb') as f:
            pickle.dump(test_tfidf, f)
        with open(os.path.join(PROCESSED_DIR, 'imdb_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        np.save(os.path.join(PROCESSED_DIR, 'imdb_train_labels_boosting.npy'), np.array(train_labels))
        np.save(os.path.join(PROCESSED_DIR, 'imdb_test_labels_boosting.npy'), np.array(test_labels))
        logger.info(f"IMDB processed data (Boosting) saved in {PROCESSED_DIR}")
    except Exception as e:
        logger.error(f"Error saving Boosting data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for mentorex2 project")
    parser.add_argument('--cifar10_vit', action='store_true', help='Process CIFAR-10 for ViT')
    parser.add_argument('--cifar10_cnn', action='store_true', help='Process CIFAR-10 for CNN')
    parser.add_argument('--imdb', action='store_true', help='Process IMDB for BERT, RNN, Boosting')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for data loaders')
    args = parser.parse_args()

    if args.cifar10_vit:
        process_cifar10_vit(args.batch_size)
    if args.cifar10_cnn:
        process_cifar10_cnn(args.batch_size)
    if args.imdb:
        process_imdb()

if __name__ == "__main__":
    main()
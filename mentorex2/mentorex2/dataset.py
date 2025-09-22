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
import pickle
import nltk

# Убедимся, что нужные NLTK-данные загружены
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Определение путей 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'mentorex2', 'data')  # Папка data в mentorex2/mentorex2/data
RAW_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Создание директорий
for directory in [INTERIM_DIR, PROCESSED_DIR]:
    os.makedirs(directory, exist_ok=True)


def process_cifar10_vit():
    """
    Preprocess CIFAR-10 dataset for ViT with batch processing.
    Saves interim and processed data.
    """
    cifar10_dir = os.path.join(RAW_DIR, 'cifar10')
    if not os.path.exists(os.path.join(cifar10_dir, 'train')) or not os.path.exists(os.path.join(cifar10_dir, 'test')):
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
    train_dataset = datasets.ImageFolder(root=os.path.join(cifar10_dir, 'train'), transform=interim_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(cifar10_dir, 'test'), transform=interim_transform)
    train_dataset_vit = datasets.ImageFolder(root=os.path.join(cifar10_dir, 'train'), transform=processed_transform_vit)
    test_dataset_vit = datasets.ImageFolder(root=os.path.join(cifar10_dir, 'test'), transform=processed_transform_vit)

    # DataLoaders для экономии памяти
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_loader_vit = DataLoader(train_dataset_vit, batch_size=batch_size, shuffle=False)
    test_loader_vit = DataLoader(test_dataset_vit, batch_size=batch_size, shuffle=False)

    # Сохранение interim данных
    with open(os.path.join(INTERIM_DIR, 'vit_cifar10_train_images.npy'), 'wb') as f_images, \
         open(os.path.join(INTERIM_DIR, 'vit_cifar10_train_labels.npy'), 'wb') as f_labels:
        for batch_images, batch_labels in train_loader:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
    with open(os.path.join(INTERIM_DIR, 'vit_cifar10_test_images.npy'), 'wb') as f_images, \
         open(os.path.join(INTERIM_DIR, 'vit_cifar10_test_labels.npy'), 'wb') as f_labels:
        for batch_images, batch_labels in test_loader:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
    print(f"CIFAR-10 interim data for ViT saved in {INTERIM_DIR}")

    # Сохранение processed данных для ViT
    with open(os.path.join(PROCESSED_DIR, 'cifar10_train_images_vit.npy'), 'wb') as f_images, \
         open(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_vit.npy'), 'wb') as f_labels:
        for batch_images, batch_labels in train_loader_vit:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
    with open(os.path.join(PROCESSED_DIR, 'cifar10_test_images_vit.npy'), 'wb') as f_images, \
         open(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_vit.npy'), 'wb') as f_labels:
        for batch_images, batch_labels in test_loader_vit:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
    print(f"CIFAR-10 processed data for ViT saved in {PROCESSED_DIR}")

def process_cifar10_cnn():
    """
    Preprocess CIFAR-10 dataset for CNN with batch processing.
    Saves processed data.
    """
    cifar10_dir = os.path.join(RAW_DIR, 'cifar10')
    if not os.path.exists(os.path.join(cifar10_dir, 'train')) or not os.path.exists(os.path.join(cifar10_dir, 'test')):
        raise FileNotFoundError(f"CIFAR-10 data not found in {cifar10_dir}. Ensure train and test folders exist.")

    # Трансформации для processed (нормализация для CNN)
    processed_transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])

    # Загрузка датасета
    train_dataset_cnn = datasets.ImageFolder(root=os.path.join(cifar10_dir, 'train'), transform=processed_transform_cnn)
    test_dataset_cnn = datasets.ImageFolder(root=os.path.join(cifar10_dir, 'test'), transform=processed_transform_cnn)

    # DataLoaders для экономии памяти
    batch_size = 1000
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=False)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=batch_size, shuffle=False)

    # Сохранение processed данных для CNN
    with open(os.path.join(PROCESSED_DIR, 'cifar10_train_images_cnn.npy'), 'wb') as f_images, \
         open(os.path.join(PROCESSED_DIR, 'cifar10_train_labels_cnn.npy'), 'wb') as f_labels:
        for batch_images, batch_labels in train_loader_cnn:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
    with open(os.path.join(PROCESSED_DIR, 'cifar10_test_images_cnn.npy'), 'wb') as f_images, \
         open(os.path.join(PROCESSED_DIR, 'cifar10_test_labels_cnn.npy'), 'wb') as f_labels:
        for batch_images, batch_labels in test_loader_cnn:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)
    print(f"CIFAR-10 processed data for CNN saved in {PROCESSED_DIR}")

def process_imdb():
    """
    Preprocess IMDB dataset for BERT, RNN, and Boosting.
    Saves interim and processed data.
    """
    imdb_path = os.path.join(RAW_DIR, 'IMDB Dataset.csv')
    if not os.path.exists(imdb_path):
        raise FileNotFoundError(f"IMDB Dataset not found at {imdb_path}")

    df = pd.read_csv(imdb_path)
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

    # Сохранение interim данных
    interim_data = {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts,
        'test_labels': test_labels
    }
    with open(os.path.join(INTERIM_DIR, 'imdb_interim.pkl'), 'wb') as f:
        pickle.dump(interim_data, f)
    print(f"IMDB interim data saved in {INTERIM_DIR}")

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
        return input_ids, attention_masks, labels

    train_input_ids, train_attention_masks, train_labels_bert = tokenize_data(train_texts, train_labels, max_length_bert)
    test_input_ids, test_attention_masks, test_labels_bert = tokenize_data(test_texts, test_labels, max_length_bert)

    torch.save(train_input_ids, os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'))
    torch.save(train_attention_masks, os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'))
    torch.save(train_labels_bert, os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'))
    torch.save(test_input_ids, os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'))
    torch.save(test_attention_masks, os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'))
    torch.save(test_labels_bert, os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'))
    print(f"IMDB processed data (BERT) saved in {PROCESSED_DIR}")

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

    torch.save(train_padded, os.path.join(PROCESSED_DIR, 'imdb_train_padded_rnn.pt'))
    torch.save(torch.tensor(train_lengths), os.path.join(PROCESSED_DIR, 'imdb_train_lengths_rnn.pt'))
    torch.save(torch.tensor(train_labels), os.path.join(PROCESSED_DIR, 'imdb_train_labels_rnn.pt'))
    torch.save(test_padded, os.path.join(PROCESSED_DIR, 'imdb_test_padded_rnn.pt'))
    torch.save(torch.tensor(test_lengths), os.path.join(PROCESSED_DIR, 'imdb_test_lengths_rnn.pt'))
    torch.save(torch.tensor(test_labels), os.path.join(PROCESSED_DIR, 'imdb_test_labels_rnn.pt'))
    with open(os.path.join(PROCESSED_DIR, 'imdb_vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)
    print(f"IMDB processed data (RNN) saved in {PROCESSED_DIR}")

    # Boosting preprocessing
    train_texts_str = [' '.join(text) for text in train_texts]
    test_texts_str = [' '.join(text) for text in test_texts]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    train_tfidf = vectorizer.fit_transform(train_texts_str)
    test_tfidf = vectorizer.transform(test_texts_str)

    with open(os.path.join(PROCESSED_DIR, 'imdb_train_tfidf.pkl'), 'wb') as f:
        pickle.dump(train_tfidf, f)
    with open(os.path.join(PROCESSED_DIR, 'imdb_test_tfidf.pkl'), 'wb') as f:
        pickle.dump(test_tfidf, f)
    with open(os.path.join(PROCESSED_DIR, 'imdb_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    np.save(os.path.join(PROCESSED_DIR, 'imdb_train_labels_boosting.npy'), np.array(train_labels))
    np.save(os.path.join(PROCESSED_DIR, 'imdb_test_labels_boosting.npy'), np.array(test_labels))
    print(f"IMDB processed data (Boosting) saved in {PROCESSED_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for mentorex2 project")
    parser.add_argument('--cifar10_vit', action='store_true', help='Process CIFAR-10 for ViT')
    parser.add_argument('--cifar10_cnn', action='store_true', help='Process CIFAR-10 for CNN')
    parser.add_argument('--imdb', action='store_true', help='Process IMDB for BERT, RNN, Boosting')
    args = parser.parse_args()

    if args.cifar10_vit:
        process_cifar10_vit()
    if args.cifar10_cnn:
        process_cifar10_cnn()
    if args.imdb:
        process_imdb()

if __name__ == "__main__":
    main()
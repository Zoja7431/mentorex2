#!/usr/bin/env python
# coding: utf-8
"""
predict_example.py - Test predictions for all models in the mentorex2 project.
"""

from mentorex2.mentorex2.modeling.train import SimpleCNN, RNNModel
from mentorex2.mentorex2.config import (
    OUTPUT_DIR_VIT, OUTPUT_DIR_CNN, OUTPUT_DIR_BERT, OUTPUT_DIR_RNN, OUTPUT_DIR_BOOSTING,
    NUM_CLASSES_CIFAR, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, MAX_LENGTH_BERT, MAX_LENGTH_RNN, TFIDF_MAX_FEATURES
)
import nltk
from nltk.tokenize import word_tokenize
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, BertForSequenceClassification, BertTokenizer
import pickle
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 class labels
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def predict_cifar10_vit(image_path):
    """Predict CIFAR-10 class using ViT model."""
    model = ViTForImageClassification.from_pretrained(OUTPUT_DIR_VIT).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image).logits
            _, predicted = torch.max(outputs, 1)
        return CIFAR10_CLASSES[predicted.item()]
    except FileNotFoundError:
        print(f"Image not found at {image_path}")
        return None


def predict_cifar10_cnn(image_path):
    """Predict CIFAR-10 class using CNN model."""
    model = SimpleCNN(num_classes=NUM_CLASSES_CIFAR).to(device)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR_CNN, 'cnn_model.pth')))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        return CIFAR10_CLASSES[predicted.item()]
    except FileNotFoundError:
        print(f"Image not found at {image_path}")
        return None


def predict_imdb_bert(text):
    """Predict IMDB sentiment using BERT model."""
    model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR_BERT, num_labels=2).to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    encodings = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LENGTH_BERT, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask).logits
        predicted = torch.argmax(outputs, dim=1).item()
    return 'positive' if predicted == 1 else 'negative'


def predict_imdb_rnn(text, rnn_type='LSTM'):
    """Predict IMDB sentiment using RNN (LSTM or GRU) model."""
    # Load vocabulary
    with open(os.path.join(OUTPUT_DIR_RNN, 'imdb_vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, rnn_type).to(device)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR_RNN, f'{rnn_type.lower()}_model.pth')))
    model.eval()

    # Tokenize and pad text
    tokens = word_tokenize(text.lower())
    token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    length = min(len(token_ids), MAX_LENGTH_RNN)
    if length == 0:
        token_ids = [vocab['<PAD>']]
        length = 1
    if len(token_ids) > MAX_LENGTH_RNN:
        token_ids = token_ids[:MAX_LENGTH_RNN]
    else:
        token_ids += [vocab['<PAD>']] * (MAX_LENGTH_RNN - len(token_ids))

    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    length_tensor = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(input_tensor, length_tensor)
        predicted = torch.argmax(outputs, dim=1).item()
    return 'positive' if predicted == 1 else 'negative'


def predict_imdb_boosting(text, model_name):
    """Predict IMDB sentiment using boosting model (XGBoost, LightGBM, CatBoost)."""
    # Load TF-IDF vectorizer
    with open(os.path.join(OUTPUT_DIR_BOOSTING, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    # Transform text
    text_tfidf = vectorizer.transform([text])

    # Load model
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier()
        model.load_model(os.path.join(OUTPUT_DIR_BOOSTING, 'xgboost_model.json'))
    elif model_name == 'LightGBM':
        model = lgb.LGBMClassifier()
        model.load_model(os.path.join(OUTPUT_DIR_BOOSTING, 'lightgbm_model.txt'))
    elif model_name == 'CatBoost':
        model = cb.CatBoostClassifier()
        model.load_model(os.path.join(OUTPUT_DIR_BOOSTING, 'catboost_model.cbm'))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    predicted = model.predict(text_tfidf)[0]
    return 'positive' if predicted == 1 else 'negative'


def main():
    """Test all models with example inputs."""
    # Example image path for CIFAR-10 (replace with actual path if available)
    image_path = r"C:\Users\Delta-Game\mentorex2\mentorex2\data\sample_cifar10_image.jpg"

    # Example text for IMDB
    example_text = "This movie was fantastic! Great acting and a compelling story."

    print("Testing ViT on CIFAR-10:")
    vit_prediction = predict_cifar10_vit(image_path)
    print(f"ViT Prediction: {vit_prediction}")

    print("\nTesting CNN on CIFAR-10:")
    cnn_prediction = predict_cifar10_cnn(image_path)
    print(f"CNN Prediction: {cnn_prediction}")

    print("\nTesting BERT on IMDB:")
    bert_prediction = predict_imdb_bert(example_text)
    print(f"BERT Prediction: {bert_prediction}")

    print("\nTesting LSTM on IMDB:")
    lstm_prediction = predict_imdb_rnn(example_text, rnn_type='LSTM')
    print(f"LSTM Prediction: {lstm_prediction}")

    print("\nTesting GRU on IMDB:")
    gru_prediction = predict_imdb_rnn(example_text, rnn_type='GRU')
    print(f"GRU Prediction: {gru_prediction}")

    print("\nTesting XGBoost on IMDB:")
    xgb_prediction = predict_imdb_boosting(example_text, 'XGBoost')
    print(f"XGBoost Prediction: {xgb_prediction}")

    print("\nTesting LightGBM on IMDB:")
    lgb_prediction = predict_imdb_boosting(example_text, 'LightGBM')
    print(f"LightGBM Prediction: {lgb_prediction}")

    print("\nTesting CatBoost on IMDB:")
    cb_prediction = predict_imdb_boosting(example_text, 'CatBoost')
    print(f"CatBoost Prediction: {cb_prediction}")


if __name__ == "__main__":
    nltk.download('punkt')
    main()
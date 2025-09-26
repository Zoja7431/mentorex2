#!/usr/bin/env python
# coding: utf-8
"""
predict.py - Test predictions for all models in the mentorex2 project.
"""

import os
import torch
import numpy as np
import pickle
from transformers import ViTForImageClassification, BertForSequenceClassification, BertTokenizer
from torchvision import transforms
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from mentorex2.mentorex2.config import (
    OUTPUT_DIR_VIT, OUTPUT_DIR_CNN, OUTPUT_DIR_BERT, OUTPUT_DIR_RNN, OUTPUT_DIR_BOOSTING,
    NUM_CLASSES_CIFAR, MAX_LENGTH_BERT, MAX_LENGTH_RNN, VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    PROCESSED_DIR
)
# from mentorex2.mentorex2.modeling.train import SimpleCNN, RNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 class labels
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Загружаем NLTK данные (запустить разово)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def predict_cifar10_vit(image_path, model_path=OUTPUT_DIR_VIT):
    """Predict CIFAR-10 class using ViT model."""
    full_path = os.path.join(model_path, 'model.safetensors')  # Унифицированный путь
    if not os.path.exists(model_path):
        print(f"ViT model directory not found at {model_path}")
        return None
    try:
        model = ViTForImageClassification.from_pretrained(model_path).to(device)
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image).logits
            _, predicted = torch.max(outputs, 1)
        return CIFAR10_CLASSES[predicted.item()]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in ViT predict: {e}")
        return None

def predict_cifar10_cnn(image_path, model_path=OUTPUT_DIR_CNN):
    """Predict CIFAR-10 class using CNN model."""
    full_path = os.path.join(model_path, 'cnn_model.pth')
    if not os.path.exists(full_path):
        print(f"CNN model file not found at {full_path}")
        return None
    try:
        model = SimpleCNN(num_classes=NUM_CLASSES_CIFAR).to(device)
        model.load_state_dict(torch.load(full_path, map_location=device, weights_only=True))
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
        ])
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        return CIFAR10_CLASSES[predicted.item()]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in CNN predict: {e}")
        return None

def predict_imdb_bert(text, model_path=OUTPUT_DIR_BERT):
    """Predict IMDB sentiment using BERT model."""
    if not os.path.exists(model_path):
        print(f"BERT model directory not found at {model_path}")
        return None
    try:
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text = re.sub(r'<br />', ' ', text.lower())
        text = re.sub(r'[^a-z ]', '', text)
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH_BERT,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_id = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(input_id, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = np.argmax(logits.detach().cpu().numpy(), axis=1).flatten()
        return "Positive" if prediction[0] == 1 else "Negative"
    except Exception as e:
        print(f"Unexpected error in BERT predict: {e}")
        return None

def predict_imdb_rnn(text, model_path=OUTPUT_DIR_RNN, rnn_type='LSTM'):
    """Predict IMDB sentiment using RNN (LSTM or GRU) model."""
    vocab_path = os.path.join(PROCESSED_DIR, 'imdb_vocab.pkl')  # Из PROCESSED_DIR, не model_path
    model_file = os.path.join(model_path, f'{rnn_type.lower()}_model.pth')
    if not os.path.exists(vocab_path):
        print(f"Vocab file not found at {vocab_path}")
        return None
    if not os.path.exists(model_file):
        print(f"{rnn_type} model file not found at {model_file}")
        return None
    try:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        model = RNNModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, rnn_type).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
        text = re.sub(r'<br />', ' ', text.lower())
        text = re.sub(r'[^a-z ]', '', text)
        tokens = [w for w in word_tokenize(text) if w not in stopwords.words('english')]
        seq = [vocab.get(word, vocab.get('<UNK>', 1)) for word in tokens]  # Добавь default для <UNK>
        seq_len = len(seq)
        if seq_len > MAX_LENGTH_RNN:
            seq = seq[:MAX_LENGTH_RNN]
        else:
            seq += [vocab.get('<PAD>', 0)] * (MAX_LENGTH_RNN - seq_len)  # Паддинг
        seq = torch.tensor([seq], dtype=torch.long).to(device)
        length = torch.tensor([min(seq_len, MAX_LENGTH_RNN)], dtype=torch.long).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(seq, length)
        prediction = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        return "Positive" if prediction == 1 else "Negative"
    except Exception as e:
        print(f"Unexpected error in {rnn_type} predict: {e}")
        return None

def predict_imdb_boosting(text, model_name='XGBoost', model_path=OUTPUT_DIR_BOOSTING):
    """Predict IMDB sentiment using boosting model (XGBoost, LightGBM, CatBoost)."""
    vectorizer_path = os.path.join(PROCESSED_DIR, 'imdb_vectorizer.pkl')  # Из PROCESSED_DIR
    if model_name == 'XGBoost':
        model_file = os.path.join(model_path, 'xgboost_model.json')
    elif model_name == 'LightGBM':
        model_file = os.path.join(model_path, 'lightgbm_model.pkl')
    elif model_name == 'CatBoost':
        model_file = os.path.join(model_path, 'catboost_model.cbm')
    else:
        print(f"Unknown model: {model_name}")
        return None
    
    if not os.path.exists(vectorizer_path):
        print(f"Vectorizer not found at {vectorizer_path}")
        return None
    if not os.path.exists(model_file):
        print(f"{model_name} model file not found at {model_file}")
        return None
    
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        if model_name == 'XGBoost':
            model = xgb.XGBClassifier()
            model.load_model(model_file)
        elif model_name == 'LightGBM':
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
        elif model_name == 'CatBoost':
            model = cb.CatBoostClassifier()
            model.load_model(model_file)
        
        cleaned_text = re.sub(r'<br />', ' ', text.lower())
        cleaned_text = re.sub(r'[^a-z ]', '', cleaned_text)
        tokens = [w for w in word_tokenize(cleaned_text) if w not in stopwords.words('english')]
        text_str = ' '.join(tokens)
        tfidf_vector = vectorizer.transform([text_str])
        pred = model.predict(tfidf_vector)[0]
        return "Positive" if pred == 1 else "Negative"
    except Exception as e:
        print(f"Unexpected error in {model_name} predict: {e}")
        return None

def main():
    """Test all models with example inputs."""
    # Сделай inputs параметрами для гибкости
    import argparse
    parser = argparse.ArgumentParser(description="Predict with mentorex2 models")
    parser.add_argument("--image_path", type=str, default="mentorex2/data/sample_cifar10_image.jpg", help="Path to CIFAR-10 image")
    parser.add_argument("--text", type=str, default="This movie was fantastic! Great acting and a compelling story.", help="IMDB text for prediction")
    args = parser.parse_args()

    image_path = args.image_path  # Относительный путь для WSL
    example_text = args.text

    print("Testing ViT on CIFAR-10:")
    vit_prediction = predict_cifar10_vit(image_path)
    print(f"ViT Prediction: {vit_prediction if vit_prediction else 'Error'}")

    print("\nTesting CNN on CIFAR-10:")
    cnn_prediction = predict_cifar10_cnn(image_path)
    print(f"CNN Prediction: {cnn_prediction if cnn_prediction else 'Error'}")

    print("\nTesting BERT on IMDB:")
    bert_prediction = predict_imdb_bert(example_text)
    print(f"BERT Prediction: {bert_prediction if bert_prediction else 'Error'}")

    print("\nTesting LSTM on IMDB:")
    lstm_prediction = predict_imdb_rnn(example_text, rnn_type='LSTM')
    print(f"LSTM Prediction: {lstm_prediction if lstm_prediction else 'Error'}")

    print("\nTesting GRU on IMDB:")
    gru_prediction = predict_imdb_rnn(example_text, rnn_type='GRU')
    print(f"GRU Prediction: {gru_prediction if gru_prediction else 'Error'}")

    print("\nTesting XGBoost on IMDB:")
    xgb_prediction = predict_imdb_boosting(example_text, 'XGBoost')
    print(f"XGBoost Prediction: {xgb_prediction if xgb_prediction else 'Error'}")

    print("\nTesting LightGBM on IMDB:")
    lgb_prediction = predict_imdb_boosting(example_text, 'LightGBM')
    print(f"LightGBM Prediction: {lgb_prediction if lgb_prediction else 'Error'}")

    print("\nTesting CatBoost on IMDB:")
    cb_prediction = predict_imdb_boosting(example_text, 'CatBoost')
    print(f"CatBoost Prediction: {cb_prediction if cb_prediction else 'Error'}")

if __name__ == "__main__":
    main()
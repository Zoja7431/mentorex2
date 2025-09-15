<<<<<<< HEAD
#!/usr/bin/env python
# coding: utf-8
"""
dataset_imdb.py - Preprocess IMDB dataset from CSV for BERT, RNN, and boosting models.
"""

import os
import sys
import pandas as pd
import torch
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from mentorex2.mentorex2.config import RAW_DIR, PROCESSED_DIR, OUTPUT_DIR_BOOSTING, MAX_LENGTH_BERT, MAX_LENGTH_RNN, VOCAB_SIZE, TFIDF_MAX_FEATURES

nltk.download('punkt')
nltk.download('stopwords')


def build_vocab(texts, vocab_size):
    """Build vocabulary from texts, limited to vocab_size."""
    word_freq = {}
    for text in texts:
        tokens = word_tokenize(text.lower())
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for i, (word, _) in enumerate(word_freq[:vocab_size - 2], 2):
        vocab[word] = i
    return vocab


def tokenize_and_pad(texts, vocab, max_length):
    """Tokenize texts and pad to max_length, returning padded sequences and lengths."""
    padded = []
    lengths = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        length = min(len(token_ids), max_length)
        if length == 0:
            token_ids = [vocab['<PAD>']]
            length = 1
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [vocab['<PAD>']] * (max_length - len(token_ids))
        padded.append(token_ids)
        lengths.append(length)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def preprocess_imdb():
    """Preprocess IMDB dataset from CSV for BERT, RNN, and boosting."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load IMDB Dataset.csv
    data_path = os.path.join(RAW_DIR, 'IMDB Dataset.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"IMDB Dataset.csv not found at {data_path}")

    df = pd.read_csv(data_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Split into train and test (50/50 as per IMDB dataset standard)
    train_df, test_df = train_test_split(df, test_size=0.5, stratify=df['sentiment'], random_state=42)
    train_texts = train_df['review'].tolist()
    train_labels = train_df['sentiment'].tolist()
    test_texts = test_df['review'].tolist()
    test_labels = test_df['sentiment'].tolist()

    # BERT preprocessing
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAX_LENGTH_BERT, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAX_LENGTH_BERT, return_tensors='pt')

    torch.save(train_encodings['input_ids'], os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'))
    torch.save(train_encodings['attention_mask'], os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'))
    torch.save(torch.tensor(train_labels, dtype=torch.long), os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'))
    torch.save(test_encodings['input_ids'], os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'))
    torch.save(test_encodings['attention_mask'], os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'))
    torch.save(torch.tensor(test_labels, dtype=torch.long), os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'))

    # RNN preprocessing
    vocab = build_vocab(train_texts, VOCAB_SIZE)
    with open(os.path.join(PROCESSED_DIR, 'imdb_vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    train_padded, train_lengths = tokenize_and_pad(train_texts, vocab, MAX_LENGTH_RNN)
    test_padded, test_lengths = tokenize_and_pad(test_texts, vocab, MAX_LENGTH_RNN)

    # Validate indices
    if train_padded.max().item() >= VOCAB_SIZE:
        raise ValueError(f"train_padded contains indices >= VOCAB_SIZE ({VOCAB_SIZE})")
    if train_padded.min().item() < 0:
        raise ValueError("train_padded contains negative indices")
    if train_lengths.min().item() <= 0:
        raise ValueError("train_lengths contains zero or negative lengths")

    torch.save(train_padded, os.path.join(PROCESSED_DIR, 'imdb_train_padded_rnn.pt'))
    torch.save(train_lengths, os.path.join(PROCESSED_DIR, 'imdb_train_lengths_rnn.pt'))
    torch.save(torch.tensor(train_labels, dtype=torch.long), os.path.join(PROCESSED_DIR, 'imdb_train_labels_rnn.pt'))
    torch.save(test_padded, os.path.join(PROCESSED_DIR, 'imdb_test_padded_rnn.pt'))
    torch.save(test_lengths, os.path.join(PROCESSED_DIR, 'imdb_test_lengths_rnn.pt'))
    torch.save(torch.tensor(test_labels, dtype=torch.long), os.path.join(PROCESSED_DIR, 'imdb_test_labels_rnn.pt'))

    # Boosting preprocessing
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    # Add feature names to avoid LightGBM warnings
    feature_names = [f'feature_{i}' for i in range(TFIDF_MAX_FEATURES)]
    train_tfidf = pd.DataFrame(train_tfidf.toarray(), columns=feature_names)
    test_tfidf = pd.DataFrame(test_tfidf.toarray(), columns=feature_names)

    with open(os.path.join(PROCESSED_DIR, 'imdb_train_tfidf.pkl'), 'wb') as f:
        pickle.dump(train_tfidf, f)
    with open(os.path.join(PROCESSED_DIR, 'imdb_test_tfidf.pkl'), 'wb') as f:
        pickle.dump(test_tfidf, f)
    with open(os.path.join(OUTPUT_DIR_BOOSTING, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    np.save(os.path.join(PROCESSED_DIR, 'imdb_train_labels_boosting.npy'), np.array(train_labels))
    np.save(os.path.join(PROCESSED_DIR, 'imdb_test_labels_boosting.npy'), np.array(test_labels))

    print("IMDB preprocessing completed.")


if __name__ == "__main__":
    preprocess_imdb()
=======
#!/usr/bin/env python
# coding: utf-8
"""
dataset_imdb.py - Preprocess IMDB dataset from CSV for BERT, RNN, and boosting models.
"""

import os
import sys
import pandas as pd
import torch
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
from mentorex2.mentorex2.config import RAW_DIR, PROCESSED_DIR, OUTPUT_DIR_BOOSTING, MAX_LENGTH_BERT, MAX_LENGTH_RNN, VOCAB_SIZE, TFIDF_MAX_FEATURES

nltk.download('punkt')
nltk.download('stopwords')


def build_vocab(texts, vocab_size):
    """Build vocabulary from texts, limited to vocab_size."""
    word_freq = {}
    for text in texts:
        tokens = word_tokenize(text.lower())
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    for i, (word, _) in enumerate(word_freq[:vocab_size - 2], 2):
        vocab[word] = i
    return vocab


def tokenize_and_pad(texts, vocab, max_length):
    """Tokenize texts and pad to max_length, returning padded sequences and lengths."""
    padded = []
    lengths = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        length = min(len(token_ids), max_length)
        if length == 0:
            token_ids = [vocab['<PAD>']]
            length = 1
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids += [vocab['<PAD>']] * (max_length - len(token_ids))
        padded.append(token_ids)
        lengths.append(length)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def preprocess_imdb():
    """Preprocess IMDB dataset from CSV for BERT, RNN, and boosting."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load IMDB Dataset.csv
    data_path = os.path.join(RAW_DIR, 'IMDB Dataset.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"IMDB Dataset.csv not found at {data_path}")

    df = pd.read_csv(data_path)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Split into train and test (50/50 as per IMDB dataset standard)
    train_df, test_df = train_test_split(df, test_size=0.5, stratify=df['sentiment'], random_state=42)
    train_texts = train_df['review'].tolist()
    train_labels = train_df['sentiment'].tolist()
    test_texts = test_df['review'].tolist()
    test_labels = test_df['sentiment'].tolist()

    # BERT preprocessing
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAX_LENGTH_BERT, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding='max_length', max_length=MAX_LENGTH_BERT, return_tensors='pt')

    torch.save(train_encodings['input_ids'], os.path.join(PROCESSED_DIR, 'imdb_train_input_ids.pt'))
    torch.save(train_encodings['attention_mask'], os.path.join(PROCESSED_DIR, 'imdb_train_attention_masks.pt'))
    torch.save(torch.tensor(train_labels, dtype=torch.long), os.path.join(PROCESSED_DIR, 'imdb_train_labels_bert.pt'))
    torch.save(test_encodings['input_ids'], os.path.join(PROCESSED_DIR, 'imdb_test_input_ids.pt'))
    torch.save(test_encodings['attention_mask'], os.path.join(PROCESSED_DIR, 'imdb_test_attention_masks.pt'))
    torch.save(torch.tensor(test_labels, dtype=torch.long), os.path.join(PROCESSED_DIR, 'imdb_test_labels_bert.pt'))

    # RNN preprocessing
    vocab = build_vocab(train_texts, VOCAB_SIZE)
    with open(os.path.join(PROCESSED_DIR, 'imdb_vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    train_padded, train_lengths = tokenize_and_pad(train_texts, vocab, MAX_LENGTH_RNN)
    test_padded, test_lengths = tokenize_and_pad(test_texts, vocab, MAX_LENGTH_RNN)

    # Validate indices
    if train_padded.max().item() >= VOCAB_SIZE:
        raise ValueError(f"train_padded contains indices >= VOCAB_SIZE ({VOCAB_SIZE})")
    if train_padded.min().item() < 0:
        raise ValueError("train_padded contains negative indices")
    if train_lengths.min().item() <= 0:
        raise ValueError("train_lengths contains zero or negative lengths")

    torch.save(train_padded, os.path.join(PROCESSED_DIR, 'imdb_train_padded_rnn.pt'))
    torch.save(train_lengths, os.path.join(PROCESSED_DIR, 'imdb_train_lengths_rnn.pt'))
    torch.save(torch.tensor(train_labels, dtype=torch.long), os.path.join(PROCESSED_DIR, 'imdb_train_labels_rnn.pt'))
    torch.save(test_padded, os.path.join(PROCESSED_DIR, 'imdb_test_padded_rnn.pt'))
    torch.save(test_lengths, os.path.join(PROCESSED_DIR, 'imdb_test_lengths_rnn.pt'))
    torch.save(torch.tensor(test_labels, dtype=torch.long), os.path.join(PROCESSED_DIR, 'imdb_test_labels_rnn.pt'))

    # Boosting preprocessing
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)

    # Add feature names to avoid LightGBM warnings
    feature_names = [f'feature_{i}' for i in range(TFIDF_MAX_FEATURES)]
    train_tfidf = pd.DataFrame(train_tfidf.toarray(), columns=feature_names)
    test_tfidf = pd.DataFrame(test_tfidf.toarray(), columns=feature_names)

    with open(os.path.join(PROCESSED_DIR, 'imdb_train_tfidf.pkl'), 'wb') as f:
        pickle.dump(train_tfidf, f)
    with open(os.path.join(PROCESSED_DIR, 'imdb_test_tfidf.pkl'), 'wb') as f:
        pickle.dump(test_tfidf, f)
    with open(os.path.join(OUTPUT_DIR_BOOSTING, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    np.save(os.path.join(PROCESSED_DIR, 'imdb_train_labels_boosting.npy'), np.array(train_labels))
    np.save(os.path.join(PROCESSED_DIR, 'imdb_test_labels_boosting.npy'), np.array(test_labels))

    print("IMDB preprocessing completed.")


if __name__ == "__main__":
    preprocess_imdb()
>>>>>>> bfff80e (Adding files)


   #!/usr/bin/env python
   # coding: utf-8
"""
test_data.py - Script to test data loading for the mentorex2 project.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mentorex2.mentorex2.features import load_cifar10_data, load_imdb_data

def main():
    print("Loading CIFAR-10 data...")
    (train_loader_vit, test_loader_vit), (train_loader_cnn, test_loader_cnn) = load_cifar10_data()
    print("CIFAR-10 data loaded successfully!")
    print("Loading IMDB data...")
    (train_loader_bert, test_loader_bert), (train_data_rnn, test_data_rnn), (X_train, X_test, y_train, y_test) = load_imdb_data()
    print("IMDB data loaded successfully!")

if __name__ == "__main__":
    main()

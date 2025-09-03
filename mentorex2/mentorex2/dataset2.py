
#!/usr/bin/env python
# coding: utf-8
"""
dataset2.py - Script to preprocess CIFAR-10 dataset for the mentorex2 project with memory-efficient batch processing.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
print("sys.path:", sys.path)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mentorex2.mentorex2.config import RAW_DIR, PROCESSED_DIR

def preprocess_cifar10():
    """Preprocess CIFAR-10 dataset for ViT and CNN with batch processing."""
    print("RAW_DIR:", RAW_DIR)
    print("PROCESSED_DIR:", PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # Transform for ViT (224x224)
    transform_vit = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])

    # Transform for CNN (32x32)
    transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261])
    ])

    # Load CIFAR-10 datasets
    print("Downloading CIFAR-10...")
    train_dataset_vit = datasets.CIFAR10(root=RAW_DIR, train=True, download=True, transform=transform_vit)
    test_dataset_vit = datasets.CIFAR10(root=RAW_DIR, train=False, download=True, transform=transform_vit)
    train_dataset_cnn = datasets.CIFAR10(root=RAW_DIR, train=True, download=True, transform=transform_cnn)
    test_dataset_cnn = datasets.CIFAR10(root=RAW_DIR, train=False, download=True, transform=transform_cnn)

    # Create DataLoaders
    batch_size = 1000  # Process 1000 images at a time to reduce memory usage
    train_loader_vit = DataLoader(train_dataset_vit, batch_size=batch_size, shuffle=False)
    test_loader_vit = DataLoader(test_dataset_vit, batch_size=batch_size, shuffle=False)
    train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=batch_size, shuffle=False)
    test_loader_cnn = DataLoader(test_dataset_cnn, batch_size=batch_size, shuffle=False)

    # Process and save ViT data
    print("Processing ViT data...")
    train_images_vit_path = os.path.join(PROCESSED_DIR, 'cifar10_train_images_vit.npy')
    train_labels_vit_path = os.path.join(PROCESSED_DIR, 'cifar10_train_labels_vit.npy')
    test_images_vit_path = os.path.join(PROCESSED_DIR, 'cifar10_test_images_vit.npy')
    test_labels_vit_path = os.path.join(PROCESSED_DIR, 'cifar10_test_labels_vit.npy')

    with open(train_images_vit_path, 'wb') as f_images, open(train_labels_vit_path, 'wb') as f_labels:
        for batch_images, batch_labels in train_loader_vit:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)

    with open(test_images_vit_path, 'wb') as f_images, open(test_labels_vit_path, 'wb') as f_labels:
        for batch_images, batch_labels in test_loader_vit:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)

    # Process and save CNN data
    print("Processing CNN data...")
    train_images_cnn_path = os.path.join(PROCESSED_DIR, 'cifar10_train_images_cnn.npy')
    train_labels_cnn_path = os.path.join(PROCESSED_DIR, 'cifar10_train_labels_cnn.npy')
    test_images_cnn_path = os.path.join(PROCESSED_DIR, 'cifar10_test_images_cnn.npy')
    test_labels_cnn_path = os.path.join(PROCESSED_DIR, 'cifar10_test_labels_cnn.npy')

    with open(train_images_cnn_path, 'wb') as f_images, open(train_labels_cnn_path, 'wb') as f_labels:
        for batch_images, batch_labels in train_loader_cnn:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)

    with open(test_images_cnn_path, 'wb') as f_images, open(test_labels_cnn_path, 'wb') as f_labels:
        for batch_images, batch_labels in test_loader_cnn:
            np.save(f_images, batch_images.numpy(), allow_pickle=False)
            np.save(f_labels, batch_labels.numpy(), allow_pickle=False)

    print("CIFAR-10 preprocessing completed!")

if __name__ == "__main__":
    preprocess_cifar10()

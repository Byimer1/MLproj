import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to normalize image data
def normalize_images(images):
    return images / 255.0

# Function to split dataset into training, validation, and test sets
def split_data(X, y, test_size=0.2, val_size=0.1):
    # First split into train+val and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Further split train into train and validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Function to apply data augmentation using Keras ImageDataGenerator
def augment_images(X_train):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X_train)
    
    return datagen

# Example of preprocessing flow for CIFAR-10
def preprocess_cifar10_data(X_train, y_train, X_test, y_test):
    # Normalize images
    X_train = normalize_images(X_train)
    X_test = normalize_images(X_test)
    
    # Split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_train, y_train, test_size=0.2, val_size=0.1)
    
    # Augment training data
    train_datagen = augment_images(X_train)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_datagen

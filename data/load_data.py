import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize pixel values
    X_train, X_test = X_train / 255.0, X_test / 255.0
    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

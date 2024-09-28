from data.load_data import load_and_preprocess_data
from models.cnn_model import build_cnn_model
from utils.preprocessing import preprocess_data
from utils.visualization import plot_accuracy, plot_loss
import tensorflow as tf


def main():
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)
    
    # Build and compile the model
    model = build_cnn_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_val, y_val),
                        batch_size=64)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
    
    # Visualize training results
    plot_accuracy(history)
    plot_loss(history)
    
    # Save the model
    model.save('image_classification_model.h5')

if __name__ == '__main__':
    main()

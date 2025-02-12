# Machine Learning Model

This project implements an image classification pipeline using a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The model is built using TensorFlow/Keras, with data preprocessing and augmentation applied using TensorFlow and scikit-learn.

## 📌 Features

- **CNN Architecture:** Uses a CNN architecture with multiple convolutional, pooling, and fully connected layers.
- **CIFAR-10 Dataset:** Processes the CIFAR-10 dataset, which consists of 60,000 images across 10 categories.
- **Data Augmentation:** Applies data augmentation to improve model generalization.
- **Data Splitting:** Splits data using scikit-learn into training, validation, and test sets.
- **Training Setup:** Trains the model with the Adam optimizer and sparse categorical cross-entropy loss.
- **Visualization:** Visualizes training progress with accuracy and loss plots.
- **Model Saving:** Saves the trained model for future use.

## 📁 Project Structure

```
MLproj/  
├── data  
│   └── load_data.py          # Loads CIFAR-10, normalizes, and splits data  
├── models  
│   └── cnn_model.py          # Defines the CNN model architecture  
├── utils  
│   ├── preprocessing.py      # Preprocessing & data augmentation utilities  
│   └── visualization.py      # Functions to plot training accuracy/loss  
└── main.py                   # Main script to train, evaluate, and save the model  
```

## ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Byimer1/MLproj.git
   cd MLproj
   ```
2. **Set up a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```
3. **Install Dependencies:**
    ```
    pip install -r requirements.txt

    ```
**If requirements.txt is missing, ensure you have the following installed:**
```
    tensorflow
    scikit-learn
    matplotlib
    numpy
```
## 🚀 Usage

To train and evaluate the model, run:

```bash
python main.py
```
### 🔹 What Happens in `main.py`?

- Loads and preprocesses the CIFAR-10 data.
- Builds and compiles the CNN model.
- Trains the model on training data.
- Evaluates model accuracy on test data.
- Visualizes training and validation performance.
- Saves the trained model to `image_classification_model.h5`.

## 🎛️ Customization

- **CNN Architecture:** Modify the architecture in `models/cnn_model.py`.
- **Data Augmentation:** Enable or adjust data augmentation in `utils/preprocessing.py`.
- **Training Parameters:** Adjust training parameters (e.g., batch size, epochs) in `main.py`.

## 📜 License

This project is open-source and licensed under the MIT License.

## 🙌 Acknowledgments

- **CIFAR-10 Dataset:** [Link](https://www.cs.toronto.edu/~kriz/cifar.html)
- **TensorFlow & Keras:** For the deep learning framework.
- **Scikit-learn:** For train-test-validation splitting.




# MNIST CNN with Pooling 

This repository contains a Convolutional Neural Network (CNN) implementation for classifying handwritten digits from the MNIST dataset.  
This version improves upon a basic CNN by including **MaxPooling** and **Dropout** layers to reduce overfitting and improve generalization.

## Features
- Uses TensorFlow/Keras for building the CNN.
- Includes **Conv2D**, **MaxPooling2D**, and **Dropout** layers.
- Achieves ~99% accuracy on the MNIST test set.
- Plots training and validation accuracy & loss.

## Model Architecture
1. Conv2D (32 filters, 3x3 kernel, ReLU)
2. MaxPooling2D (2x2)
3. Dropout (0.25)
4. Conv2D (32 filters, 3x3 kernel, ReLU)
5. MaxPooling2D (2x2)
6. Dropout (0.25)
7. Flatten
8. Dense (128, ReLU)
9. Dense (10, Softmax)

## Usage

from tensorflow.keras.models import load_model

model = load_model("MNIST_CNN_MODEL.h5", compile=False)

# Predict
predictions = model.predict(X_test)

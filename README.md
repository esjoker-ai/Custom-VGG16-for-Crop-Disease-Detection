# Custom-VGG16-for-Crop-Disease-Detection
This project implements a custom deep learning model for detecting crop diseases from images. The solution uses a custom-designed VGG16-style architecture, rather than a pre-built model, as its base. The model is then fine-tuned for the specific task of crop disease classification. This notebook, agriculture_without_aug_final.ipynb, outlines the complete process from data loading and preprocessing to model training and evaluation.

Methodology
Data Loading and Preprocessing:

Images are loaded from a Google Drive directory named combined_dataset.

All images are resized to a uniform size of 224x224 pixels.

Labels for each image are extracted from the directory structure and then one-hot encoded for model compatibility.

The dataset is split into training and testing sets with a 70/30 ratio.

Pixel values are normalized to a range of 0 to 1.

Model Architecture:

A custom VGG16-style convolutional base is implemented, with new layers added on top, including a GlobalAveragePooling2D layer, a Dense layer with 512 neurons and a relu activation, BatchNormalization, Dropout for regularization, and a final Dense layer with softmax activation for classification.

Training and Evaluation:

The model is compiled with the Adam optimizer and a learning rate of 0.0001.

The model is trained for 50 epochs.

Validation is performed on the test data to monitor performance and prevent overfitting.

Prerequisites
Python 3.x

Jupyter Notebook

TensorFlow

Keras

NumPy

scikit-learn

OpenCV

Matplotlib

The combined_dataset in your Google Drive, mounted in Colab.

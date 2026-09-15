# CodeAlpha Handwritten Character Recognition

## CodeAlpha Task 3

An AI-based handwritten digit recognition system developed using Python, TensorFlow, CNN, and Streamlit.

## Project Objective

The objective of this project is to identify handwritten digits from 0 to 9 using a Convolutional Neural Network (CNN).

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- Convolutional Neural Network (CNN)

## Dataset

The project uses the MNIST handwritten digit dataset.

The dataset contains grayscale images of handwritten digits from 0 to 9.

Each image is converted into a 28 × 28 pixel format before being provided to the CNN model.

## Key Features

- Upload handwritten digit images
- Automatic image preprocessing
- CNN-based digit classification
- Prediction from 0 to 9
- Confidence score
- Prediction probability for every digit
- Interactive Streamlit interface

## Project Structure

CodeAlpha_HandwrittenCharacterRecognition/

├── app.py

├── train_model.py

├── requirements.txt

├── README.md

└── models/

    └── mnist_cnn.keras

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
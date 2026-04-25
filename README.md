# CIFAR-10 Classifier with PyTorch

This project implements a simple Convolutional Neural Network for CIFAR-10 image classification using PyTorch.

## Dataset

The CIFAR-10 dataset contains 60,000 colour images in 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Model

The model is a simple CNN with:

- 2 convolutional layers
- 2 max pooling layers
- 2 fully connected layers

## Project Structure

```text
cifar10-classifier-pytorch/
├── models/
│   └── cnn.py
├── results/
│   ├── loss_curve.png
│   └── accuracy_curve.png
├── saved_models/
│   └── best_model.pth
├── train.py
├── requirements.txt
└── README.md
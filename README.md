# ADA-GP: Accelerating DNN Training with Adaptive Gradient Prediction

## Overview

This project implements **ADA-GP (Adaptive Gradient Prediction)**, a novel approach to accelerate the training of Deep Neural Networks (DNNs) while maintaining high accuracy. ADA-GP utilizes an auxiliary neural network to predict gradients during training, alternating between backpropagation (BP) and gradient prediction (GP). This methodology allows for faster training times by reducing reliance on traditional backpropagation, particularly for large models and datasets.

## Features

- **Gradient Prediction**: Introduces a secondary model to predict gradients, reducing the computational cost of gradient calculation.
- **Adaptive Training Phases**: Alternates between standard backpropagation and gradient prediction to optimize the learning process.
- **Efficiency**: Aims to speed up DNN training without compromising on model accuracy.
- **Visualization**: Includes visualizations of training, testing, and predicted data to track model performance over time.


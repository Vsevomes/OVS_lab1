# OVS_lab1: Neural Network Implementation

This repository contains a C++ implementation of a neural network designed for pattern recognition tasks. The project is structured to facilitate both training and inference operations, with support for custom datasets.

## Features

- **Custom Neural Network**: Implemented from scratch in C++.
- **Training and Inference**: Capable of training on labeled datasets and performing inference on new data.
- **Dataset Generation**: Includes scripts for generating training and testing datasets in a specific format.
- **Evaluation Metrics**: Outputs training time, number of epochs, inference time, and memory usage during training and inference.

## Project Structure

```
OVS_lab1/
├── CMakeLists.txt # Build configuration
├── main.cpp # Main application entry point
├── include/ # Header files
│ └── neural_network.hpp # Neural network class and utilities
├── src/ # Source files
│ └── neural_network.cpp # Implementation of neural network
├── data/ # Dataset files
│ ├── train/ # Training samples
│ └── test/ # Test samples
└── README.md # Project documentation
```

## Requirements

- C++17 or later
- CMake 3.10 or later
- Python 3.x (for dataset generation)

## Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/Vsevomes/OVS_lab1.git
   cd OVS_lab1
   ```

2. Create a build directory and compile:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
3. The build will produce an executable file (e.g., neural_network) in the build directory.

## Usage

You can use the executable to either train a model or predict using a saved model.

1. Train a Model

    ```bash
    ./neural_network train <train_folder> <num_inputs> <hiddenLayers> <neuronsPerLayer> <num_outputs> <learningRate>
    ```
    - <train_folder> – folder containing training files
    - <num_inputs> – number of input neurons (e.g., 49 for 7×7 images)
    - <hiddenLayers> – number of hidden layers
    - <neuronsPerLayer> – number of neurons per hidden layer
    - <num_outputs> – number of output neurons/classes
    - <learningRate> – learning rate for training

    Example:
    ```bash
    ./neural_network train ../data/train 49 1 20 3 0.1
    ```
2. Predict Using a Saved Model
    ```bash
    ./neural_network predict <model_file> <input_file>
    ```
    - <model_file> – path to the saved model file
    - <input_file> – path to a file containing a single input shape

    Example:
    ```bash
    ./neural_network predict model.dat ../data/test/circle_0.txt
    ```

## Dataset Format

Each dataset file should contain:

1. The first line: a space-separated vector representing the one-hot encoded target class.
2. Subsequent lines: a 7x7 grid of binary values representing the input pattern.

Example:
```
0 0 0 0 0 0 0
0 0 0 1 0 0 0
0 0 1 0 1 0 0
0 1 0 0 0 1 0
1 0 0 0 0 0 1
1 1 1 1 1 1 1
0 0 0 0 0 0 0
```

## Generating Datasets

Python scripts are provided to generate training and testing datasets:
```bash
python generate_datasets.py
```

This will create train/ and test/ directories populated with sample files.

## Evaluation Metrics

Upon execution, the program will output:

- Training Time: Duration of the training process.
- Epoch Count: Number of epochs completed.
- Inference Time: Time taken to process the test dataset.
- Memory Usage: Estimated memory consumption during training and inference.
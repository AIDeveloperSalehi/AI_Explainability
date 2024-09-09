# MNIST Explainability Study

This project demonstrates the importance of explainability in machine learning models using the MNIST dataset. It showcases how a CNN model can be vulnerable to simple modifications in input data, emphasizing the need for thorough model analysis beyond accuracy metrics.

## Project Structure

- `Train_CNN_Normal_MNIST.ipynb`: Trains a CNN on the standard MNIST dataset.
- `Train_CNN_Framed_MNIST.ipynb`: Trains a CNN on a modified MNIST dataset where digit 9 is framed.
- `Explainability.ipynb`: Analyzes the trained models using Grad-CAM for explainability.
- `download_MNIST.py`: Script to download the MNIST dataset.
- `Utils.py`: Contains utility functions and model definitions.

## Key Features

1. Standard MNIST classification using CNN
2. Modified MNIST dataset with framed digit 9
3. Comparison of model performance on original and modified datasets
4. Explainability analysis using Grad-CAM
5. Visualization of model decision-making process

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/AIDeveloperSalehi/AI_Explainability.git
   cd AI_Explainability
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the MNIST dataset:
   ```
   python download_MNIST.py
   ```

## Usage

1. Run `Train_CNN_Normal_MNIST.ipynb` to train the model on the standard MNIST dataset.
2. Run `Train_CNN_Framed_MNIST.ipynb` to train the model on the modified MNIST dataset.
3. Use `Explainability.ipynb` to analyze the trained models and visualize their decision-making process.

## Results

The project demonstrates how a high-performing model on standard metrics can have significant vulnerabilities. By framing digit 9 in the dataset, we show that the model learns to associate the frame with the digit, leading to misclassifications when frames are added to other digits.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

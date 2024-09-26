# **Classification of Noisy MNIST Images Using Deep Learning**

This project aims to classify noisy images of handwritten digits from the MNIST dataset using various deep learning models. The project explores different architectures, including **Multi-Layer Perceptrons (MLP)** and **Convolutional Neural Networks (CNN)**, to determine the most accurate model for this task.

## **Project Structure**
The project is implemented in a Jupyter notebook:

- `Projet_MNIST.ipynb` - Main notebook containing the data preprocessing, model development, training, and evaluation steps.

## **Objective**
The primary objective of this project is to accurately classify noisy MNIST images using deep learning models. The noise added to the images simulates a more challenging real-world environment where input data might be noisy or distorted.

## **Approach**
We experimented with two main types of neural network architectures:

- **MLP (Multi-Layer Perceptron)**: A fully connected network that takes the flattened pixel data as input.
- **CNN (Convolutional Neural Network)**: A convolutional architecture that processes the spatial structure of the image for feature extraction.

### Key Models:
1. **MLP (Fully Connected Networks)**: Various architectures with hidden layers and dropout for regularization.
2. **CNN (Convolutional Neural Networks)**: Different CNN architectures with convolutional layers followed by max-pooling, flattening, and dense layers.

## **Data Preprocessing**
- The MNIST dataset is used, with additional noise added to simulate real-world scenarios.
- **Normalization**: The pixel values are normalized to a range of [0, 1] for faster convergence during training.
- **One-hot encoding**: The labels are one-hot encoded for categorical classification.

## **Model Performance**
After testing multiple architectures, the **CNN model** showed the best performance on both the training set and during cross-validation.

- **Best Model**: CNN with the following performance:
  - **Accuracy**: 96.46%
  - **Loss**: 0.1569
  - **Cross-validation**: 96.62% (+/- 0.25%)

## **Model Architecture**
### CNN (Best Performing Model):
- **Input Layer**: Image input with dimensions 28x28.
- **Convolutional Layers**: Two or more Conv2D layers with ReLU activation and MaxPooling.
- **Fully Connected Layers**: Dense layers with ReLU activation and dropout for regularization.
- **Output Layer**: Softmax for multi-class classification (digits 0-9).

The CNN architecture outperformed the MLP models due to its ability to capture spatial hierarchies in the image data, which are crucial for digit classification.

## **Evaluation**
The models were evaluated using **cross-validation** to ensure robust performance across different subsets of the data.

- **Cross-validation accuracy**: 96.62% with a standard deviation of +/- 0.25%.
- **Accuracy** and **Loss** were recorded for each epoch to monitor training progress.

## **Installation & Usage**
To run this project, follow these steps:

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/theo-serralta/MNIST_classifier.git
    ```

2. Navigate to the project directory:
    ```bash
    cd MNIST_classifier
    ```

3. Open the Jupyter notebook:
    ```bash
    jupyter notebook Projet_MNIST.ipynb
    ```

## **Requirements**
Make sure the following dependencies are installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## **Conclusion**
This project demonstrates the power of **CNNs** for image classification tasks, especially when dealing with noisy data. The CNN model achieved a high accuracy of **96.46%** on noisy MNIST images, outperforming the MLP models. The use of convolutional layers allowed the model to capture important features, leading to better generalization and robustness in the presence of noise.

## **Future Work**
Potential improvements and future directions include:
- Experimenting with deeper CNN architectures such as **ResNet** or **DenseNet**.
- Using advanced data augmentation techniques to further improve robustness.
- Applying the model to other noisy datasets to test generalizability.

---

## **Author**
Theo Serralta
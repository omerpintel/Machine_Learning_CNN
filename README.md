CNN-Based Shoe Pair Classification
Overview
This project involves building a Convolutional Neural Network (CNN) that predicts whether two shoes form a matching pair or if they are from different pairs. This can be particularly useful in real-world applications, such as providing assistance to visually impaired individuals for identifying pairs of shoes.

The project explores two different CNN architectures. Although there is starter code for data processing, the model-building and training parts of this assignment allow a good deal of freedom for experimentation.

Files
The following files are necessary to run and evaluate the models:

ML_DL_Assignment3.ipynb: The main notebook containing the model-building process, training, and evaluation.
ML_DL_Functions3.py: Helper functions for the data preprocessing and model training.
best_CNN_model.pk: The parameters for the best-performing CNN model (architecture 1).
best_CNNChannel_model.pk: The parameters for the best-performing CNN model with modified channels (architecture 2).
Dataset
The dataset includes images of shoes, divided into the following directories:

train: Images used for training and validation.
test_w: Test set containing images of women's shoes.
test_m: Test set containing images of men's shoes.
The dataset consists of triplets of shoe pairs, where each triplet contains images taken under similar conditions. The goal is to use these images to train models capable of classifying whether the shoes are a matching pair or not.

Project Structure
Data Preprocessing: The dataset is preprocessed to ensure consistency between the training and test sets. This includes resizing images, normalizing pixel values, and creating data loaders.

Model Architecture: Two CNN architectures are explored. The first model is a basic CNN, while the second model adds modifications to the channel handling. Both models are built using PyTorch.

Training and Evaluation: The models are trained on the training set and evaluated on the test set using accuracy as the main metric. A validation set is used during training to ensure that the model does not overfit.

Results: After training, the performance of each model is compared on both men's and women's shoe test sets to evaluate their effectiveness.

Dependencies
Python 3.x
PyTorch
NumPy
Matplotlib
Google Colab (for training in the cloud)

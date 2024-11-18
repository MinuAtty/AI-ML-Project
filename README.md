# AI-ML-Project

## Scenario overview: 

As people increasingly use digital devices like smartphones and smartwatches to access sensitive information, such as online payments and banking, it's crucial to have an enhanced authentication method that continuously and transparently safeguards user data from unauthorized access. This assessment seeks to evaluate acceleration-based features and explore their potential for verifying user identity through neural networks.



### Step 1: Understanding the Problem and Dataset

Objective:
- Develop a neural network-based system for user authentication using acceleration-based features. 
- Train a Feedforward Multi-Layer Perceptron (MLP) to classify users based on movement data.

Dataset Structure:
- Each user has multiple feature sets derived from time and frequency domain acceleration data. 
- Files contain data for inter- and intra-user sessions, representing variations in movement patterns.



### Step 2: Data Loading and Preprocessing

Load the Dataset:
- Import .mat files for each user and concatenate the data for each user.
- Assign labels to the samples based on the user ID.

Check Data Consistency:
- Verify that all files for a user have the same number of columns (features).
- Handle any inconsistencies by truncating or padding where necessary.

Normalize Features:
- Normalize the feature values to bring all features to the same scale (e.g., between 0 and 1).

Split the Dataset:
- Split the dataset into training (80%) and test (20%) sets.
- Use stratified splitting to ensure an even distribution of user data across both sets.



### Step 3: Perform Data Analysis

Descriptive Statistics:
- Compute summary statistics (mean, standard deviation, variance) for the entire dataset.
- Analyze intra-user variance (variations within a single user) and inter-user variance (variations between users).
- Visualize the data using plots such as histograms, box plots, or scatter plots to identify patterns or anomalies.

Feature Correlation Analysis:
- Analyze correlations between features to understand which ones contribute the most to user differentiation.



### Step 4: Build and Train the Neural Network

Design the Neural Network:
- Use MATLAB’s feedforwardnet to define a Feedforward Multi-Layer Perceptron (MLP) with:
- Input layer: Number of features in the dataset.
- Hidden layers: Experiment with different configurations (e.g., 3 layers with 50, 25, and 10 neurons).
- Output layer: Number of users (one neuron per user, with softmax activation).

Configure the Training Parameters:
- Learning algorithm: Use the Levenberg-Marquardt backpropagation (trainlm).
- Learning rate: Set an initial value (e.g., 0.01) and adjust during optimization.
- Epochs: Use 100 epochs to train the model.

Train the Model:
- Use the training data to train the neural network.
- Monitor training and validation accuracy to detect overfitting.



### Step 5: Evaluate the Model

Evaluate Test Performance:
- Use the test data to evaluate the model's accuracy.
- Generate a confusion matrix to analyze misclassifications.

Calculate Metrics:
- Precision, Recall, and F1-Score to assess the classification performance.
- Analyze these metrics for individual users to identify weak points in the model.

Visualize Results:
- Plot training and validation accuracy/loss curves to understand the model's learning behavior.



### Step 6: Optimize Model Performance

Feature Selection:
- Apply techniques like Principal Component Analysis (PCA) to reduce dimensionality.
- Retain features that capture the most variance (e.g., 95%).

Hyperparameter Tuning:
- Experiment with different learning rates, number of neurons, activation functions, or batch sizes.
- Use grid search or random search methods to find the best hyperparameter combination.

Classifier Comparison (Optional):
- Compare MLP performance with other classifiers like Support Vector Machines (SVM) or k-Nearest Neighbors (kNN).

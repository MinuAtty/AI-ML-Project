# AI-ML-Project

## Scenario overview: 

As people increasingly use digital devices like smartphones and smartwatches to access sensitive information, such as online payments and banking, it's crucial to have an enhanced authentication method that continuously and transparently safeguards user data from unauthorized access. This assessment seeks to evaluate acceleration-based features and explore their potential for verifying user identity through neural networks.


## 1. Data Analysis and Preparation

### Step 1: Understand the Dataset
- Review the provided dataset and feature descriptions.
- Take note of:
The distinction between time and frequency domain features.
How data differs across "same-day" and "cross-day" datasets.

### Step 2: Perform Descriptive Statistics
- Intra-user variance: Analyze the variance of features within each user's data (e.g., for "same-day" and "cross-day" scenarios).
- Inter-user variance: Examine differences in feature distributions between users.
- Use the following techniques:
Calculate means, standard deviations, and correlations.
Create visualizations (e.g., box plots, histograms, pair plots).
Perform Principal Component Analysis (PCA) or t-SNE to visualize feature separability.

### Step 3: Normalize and Standardize the Data
- Normalize feature values for comparability (e.g., scale features to 0-1 or use z-score normalization).
- Check for missing values or outliers, and address them appropriately.

## 2. Neural Network Modeling

### Step 4: Prepare the Data for Training
- Combine data from all users into a unified dataset, labeling samples by user ID.
- Split the data into:
Training set (e.g., 70%)
Validation set (e.g., 15%)
Test set (e.g., 15%)

### Step 5: Configure the Feedforward Neural Network
- Use a Feedforward Multi-Layer Perceptron (MLP), implemented in Python (e.g., TensorFlow, PyTorch) or MATLAB (feedforwardnet).
- Suggested configuration:
Input layer: 131 neurons (if using combined features).
Hidden layers: Start with 2-3 layers with 64, 128, or 256 neurons each.
Output layer: 10 neurons (one for each user, using softmax for classification).
Activation functions: Use ReLU for hidden layers and softmax for output.
Loss function: Cross-entropy for classification.
Optimizer: Adam or SGD with learning rate tuning.
- Train the model for a sufficient number of epochs (e.g., 50-200), using early stopping.

### Step 6: Evaluate the Model
- Measure model performance using:
Accuracy
Precision, Recall, and F1-Score (for individual users).
Confusion matrix.
- Visualize loss and accuracy curves to assess overfitting or underfitting.

## 3. Feature Optimization

### Step 7: Feature Selection
- Experiment with subsets of features:
Time domain only (88 features).
Frequency domain only (43 features).
Combined domain features (131 features).
- Use feature importance metrics (e.g., feature weights from a trained model, Recursive Feature Elimination (RFE), or PCA).

### Step 8: Fine-tune Classifier
- Experiment with MLP configurations:
Adjust the number of hidden layers, neurons, and activation functions.
Apply dropout or batch normalization to improve generalization.
- Optimize hyperparameters using:
Grid Search
Random Search
Bayesian Optimization

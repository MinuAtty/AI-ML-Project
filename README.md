# Artificial Intelligence & Machine Learning Coursework

## Scenario overview: 

As people increasingly use digital devices like smartphones and smartwatches to access sensitive information, such as online payments and banking, it's crucial to have an enhanced authentication method that continuously and transparently safeguards user data from unauthorized access. This assessment seeks to evaluate acceleration-based features and explore their potential for verifying user identity through neural networks.

## Dataset Information: 

The acceleration-based feature sets were generated for ten users in total. For each user, a set of time and frequency domain features was created, comprising 131 features in total—88 features from the time domain and 43 from the frequency domain. This process produced six feature vectors for each user. The 
details of these feature vectors are outlined below in  feature vectors information: 


|  File name                 |     Description                                                                                |   NF    |   NS   |
|  ------------------------- |    -----------------------------------------------------------------------------------------   |  -----  |  ----  |
|  U01_Acc_FreqD_FDay        |    Frequency domain-based acceleration features for User 1 – using the same-day data           |   43    |   36   |
|  U01_Acc_TimeD_FDay        |    Time domain-based acceleration features for User 1- using the cross-day data                |   88    |   36   |    
|  U01_Acc_FreqD_MDay        |    Frequency domain-based acceleration features for User 1- using the cross-day data           |   43    |   36   |
|  U01_Acc_TimeD_MDay        |    Time domain-based acceleration features for User 1- using the cross-day data                |   88    |   36   |
|  U01_Acc_TimeD_FreqD_FDay  |    Time and frequency domain-based acceleration features for User 1- using the same-day data   |   131   |   36   |
|  U01_Acc_TimeD_FreqD_MDay  |    Time and frequency domain-based acceleration features for User 1- using the same-day data   |   131   |   36   |

- U01: User with its associated ID 
- Acc: Acceleration-based data 
- FreqD: Frequency domain-based Features   
- TimeD: Time domain-based Features
- FDay: First day Features
- MDay: Cross-day Features 
- NF: Number of features 
- NS: Number of samples

<br>

The provided MATLAB script implements a neural network model for a user recognition task using user-specific features. Below are the steps organized for clarity:

---

### **1. Initialization**
1. **Hyperparameters Definition**:
   - Define a grid of hidden layer sizes (`hidden_layer_sizes`) and learning rates (`learning_rates`) for grid search.
   - Initialize variables for overall accuracy, labels storage, and user-specific metrics (intra-variance).

2. **Storage Preparation**:
   - Initialize arrays and structures to hold results, accuracy values, and combined user data for inter-user variance calculation.

---

### **2. Dataset Preparation**
1. **User-Specific Data Loading**:
   - Loop through each user (`currentUserNum`) and load their dataset (both "F-Day" and "M-Day") from `.mat` files.
   - Combine features from both datasets and assign labels (`1` for the target user, `0` for others).

2. **Variance Calculation**:
   - **Intra-variance**:
     - Compute the coefficient of variation (normalized standard deviation) for each user’s features.
   - **Inter-variance**:
     - Compute feature-level standard deviations and means across all users, normalize, and average them.

---

### **3. Data Preprocessing**
1. **Centering and Correlation Analysis**:
   - Center the data (subtract the mean).
   - Compute the correlation matrix to analyze linear dependencies and handle numerical issues.

2. **Dimensionality Reduction**:
   - Perform eigenvalue decomposition to retain only significant components (eigenvalues above a threshold).
   - Project data onto the reduced components using PCA.

---

### **4. Data Splitting**
1. **Training and Testing Split**:
   - Split the dataset into training (60%) and testing (40%) subsets.

2. **Validation Split**:
   - Further split the training data into a final training set and a validation set (20% validation).

---

### **5. Neural Network Training with Grid Search**
1. **Grid Search for Hyperparameters**:
   - Iterate through all combinations of `hidden_layer_sizes` and `learning_rates`.
   - For each combination:
     - Train a feedforward neural network using the current parameters.
     - Evaluate its performance on the validation set.
   - Store results and update the best-performing model parameters.

2. **Final Training**:
   - After identifying the optimal parameters, train the best network on the final training set.

---

### **6. Model Evaluation**
1. **Training Accuracy**:
   - Evaluate the model on the training set and calculate accuracy.

2. **Testing Accuracy**:
   - Predict labels for the testing set and calculate accuracy.
   - Update overall accuracy and accumulate true and predicted labels for all users.

---

### **7. Performance Metrics**
1. **Grid Search Results**:
   - Display the grid search results (hidden neurons, learning rate, and accuracy) for each user.

2. **Combined Results**:
   - Calculate and visualize mean and standard deviation of validation accuracies across all users as a heatmap.

3. **Variance Metrics**:
   - Display intra-variance (per user) and inter-variance (across users).

4. **Accuracy Metrics**:
   - Calculate the average accuracy across all users.

---

### **8. Visualization**
1. **Confusion Matrices**:
   - Plot confusion matrices for training and testing data.

2. **User Statistics**:
   - Plot bar charts for:
     - Mean feature values per user.
     - Standard deviation of features per user.
     - Intra-variance per user.

---

### **9. Final Report**
- The script provides comprehensive performance reporting through:
  - Grid search results for all users.
  - Heatmaps of accuracies.
  - Variance calculations.
  - Accuracy summaries.
  - Confusion matrices for classification performance.

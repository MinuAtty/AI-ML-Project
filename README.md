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

### **1. Data Loading and Preparation**
1. **Load User Data**:
   - Load acceleration-based feature datasets for 10 users (`*_Acc_FreqD_FDay.mat` and `*_Acc_FreqD_MDay.mat`).
   - Extract feature matrices (`Acc_FD_Feat_Vec`) for frequency and time-domain data.

2. **Label Assignment**:
   - Assign labels:
     - **1** for the target user's data.
     - **0** for other users' data.

3. **Combine Data**:
   - Concatenate all feature matrices for each user into a single dataset (`user_data`).
   - Store labels in `user_labels`.

---

### **2. Variance Calculations**
1. **Intra-Variance**:
   - Compute the standard deviation of features for each user.
   - Normalize by dividing by the maximum feature value for stability.
   - Average the normalized standard deviations across features for intra-variance per user.

2. **Inter-Variance**:
   - Calculate the mean and standard deviation of features across all users.
   - Compute the ratio of feature-wise standard deviation to mean to assess inter-variance.

---

### **3. Principal Component Analysis (PCA)**
1. **Center the Data**:
   - Subtract the mean from the feature data to center it around zero.

2. **Compute Correlation Matrix**:
   - Generate the correlation matrix of the centered data.

3. **Eigenvalue Decomposition**:
   - Decompose the correlation matrix into eigenvalues and eigenvectors.
   - Retain significant components based on eigenvalues (`>1e-10`).

4. **Project Data**:
   - Transform the original data into the reduced PCA space.

---

### **4. Data Splitting**
1. **Train-Test Split**:
   - Use 60% of the data for training and 40% for testing (`cvpartition`).
   - Split both data (`user_data_pca`) and labels (`user_labels`).

2. **Train-Validation Split**:
   - Split the training data into:
     - **Training set (80%)** for fitting the model.
     - **Validation set (20%)** for hyperparameter tuning.

---

### **5. Neural Network Configuration**
1. **Define Feedforward Neural Network**:
   - Use MATLAB's `feedforwardnet` with adjustable hidden layer sizes.

2. **Set Parameters**:
   - Number of neurons in hidden layers: `[10, 20, 30]`.
   - Learning rates: `[0.01, 0.1]`.
   - Maximum epochs: `1000`.

3. **Grid Search for Hyperparameters**:
   - Train networks with different combinations of hidden neurons and learning rates.
   - Evaluate performance on the validation set using binary classification accuracy.

4. **Select Optimal Model**:
   - Identify the network configuration with the highest validation accuracy.

---

### **6. Training and Testing**
1. **Train Final Model**:
   - Use the optimal hyperparameters to train the model on the final training set.

2. **Test Model**:
   - Predict labels for the test set using the trained neural network.
   - Convert raw predictions to binary labels using a threshold (≥0.5).

---

### **7. Evaluation**
1. **Calculate Accuracy**:
   - Compute accuracy for both training and testing datasets.

2. **Confusion Matrices**:
   - Generate confusion matrices for training and testing predictions to visualize classification performance.

---

### **8. Results and Optimization**
1. **Optimization**:
   - Compare accuracy and variance metrics before and after applying PCA.
   - Assess the impact of hyperparameter tuning on model performance.

2. **Visualization**:
   - Plot:
     - Intra-variance per user.
     - Inter-variance across all users.
     - Confusion matrices for training and testing.

clc; clear; close all;

% Define hyperparameters globally
hidden_layer_sizes = [10, 20, 30]; % Different numbers of hidden neurons
learning_rates = [0.01, 0.1]; % Different learning rates

overall_accuracy = 0; % Initialize overall accuracy
true_labels = []; % Initialize array to accumulate true labels
predicted_labels = []; % Initialize array to accumulate predictions

% Initialize storage arrays
user_intra_variance = zeros(10, 1);
combined_user_data = [];
all_results = cell(10, 1);
all_accuracies = zeros(length(hidden_layer_sizes), length(learning_rates), 10);

for currentUserNum = 1:10
    currentUserID = sprintf('U%02d', currentUserNum); % Format user ID as U01, U02, ..., U10

    % Initialize per-user data and labels
    user_data = [];
    user_labels = [];

    % Loop through all users to prepare dataset
    for userNum = 1:10
        userID = sprintf('U%02d', userNum);

        % Load datasets for F-day and M-day for each user
        try
            data_FDay = load(fullfile("Data", userID + "_Acc_FreqD_FDay.mat"));
            data_MDay = load(fullfile("Data", userID + "_Acc_FreqD_MDay.mat"));
        catch
            warning(['Data file not found for user: ', userID]);
            continue; % Skip if data is missing
        end

        % Extract feature matrices
        features_FDay = data_FDay.Acc_FD_Feat_Vec;
        features_MDay = data_MDay.Acc_FD_Feat_Vec;

        % Merge data row-wise
        user_data = [user_data; features_FDay; features_MDay];

        % Assign labels: 1 for target user, 0 for others
        if userNum == currentUserNum
            user_labels = [user_labels; ones(size(features_FDay, 1), 1); ones(size(features_MDay, 1), 1)];
        else
            user_labels = [user_labels; zeros(size(features_FDay, 1), 1); zeros(size(features_MDay, 1), 1)];
        end

        % Calculate intra-variance for the current user
        current_user_data = [features_FDay; features_MDay];
        current_user_mean = mean(current_user_data, 1);
        current_user_std = std(current_user_data, 0, 1);
        
        % Calculate coefficient of variation with handling for edge cases
        max_values = max(abs(current_user_data), [], 1);
        normalized_std = current_user_std ./ max_values;  % Normalize by max value instead of mean
        % Remove any remaining NaN or Inf values
        normalized_std(isnan(normalized_std) | isinf(normalized_std)) = 0;
        user_intra_variance(userNum) = mean(normalized_std);

        % Accumulate data for inter-variance calculation
        combined_user_data = [combined_user_data; current_user_data];
    end
     %rng(1);
    % Suppress PCA warnings
    warning('off', 'MATLAB:singularMatrix');
    warning('off', 'MATLAB:rankDeficientMatrix');

    % Preprocess data to handle linear dependencies
    % Center the data
    user_data = user_data - mean(user_data);
    
    % Compute correlation matrix
    correlation_matrix = corrcoef(user_data);
    
    % Handle numerical issues
    correlation_matrix(isnan(correlation_matrix)) = 0;
    correlation_matrix(abs(correlation_matrix) < 1e-10) = 0;
    
    % Eigenvalue decomposition of correlation matrix
    [eigen_vectors, eigen_values] = eig(correlation_matrix);
    eigen_values_diag = diag(eigen_values);
    
    % Keep only components with significant eigenvalues
    tolerance = 1e-10;
    significant_components = abs(eigen_values_diag) > tolerance;
    reduced_eigen_vectors = eigen_vectors(:, significant_components);
    
    % Project data onto significant components
    user_data_pca = user_data * reduced_eigen_vectors;

    % Split data into training and testing sets
    cv_partition = cvpartition(size(user_data_pca, 1), 'HoldOut', 0.4); % 60% training, 40% testing
    train_indices = training(cv_partition);
    test_indices = test(cv_partition);

    % Use PCA-transformed data for training and testing
    train_data = user_data_pca(train_indices, :);
    test_data = user_data_pca(test_indices, :);
   
    % Labels for training and testing
    y_train = user_labels(train_indices);
    y_test = user_labels(test_indices);

    % Hyperparameter Grid Search
    optimal_accuracy = 0;
    optimal_params = struct();
    
    % Create validation set
    cv_val_partition = cvpartition(length(y_train), 'HoldOut', 0.2);
    val_indices = test(cv_val_partition);
    final_train_data = train_data(~val_indices, :);
    val_data = train_data(val_indices, :);
    final_y_train = y_train(~val_indices);
    y_val = y_train(val_indices);
    
    % Grid Search
    % Initialize results table
    results_table = zeros(length(hidden_layer_sizes) * length(learning_rates), 3);
    row = 1;
    
    for hidden_size = hidden_layer_sizes
        for lr = learning_rates
            % Configure network with current parameters
            net = feedforwardnet(hidden_size);
            net.trainParam.epochs = 500;
            net.trainParam.goal = 1e-6;
            net.trainParam.max_fail = 6;
            net.trainParam.lr = lr;
            
            % Train network
            [trial_net, ~] = train(net, final_train_data', final_y_train');
            
            % Evaluate on validation set
            val_pred = trial_net(val_data');
            val_pred_binary = double(val_pred >= 0.5)';
            val_accuracy = sum(val_pred_binary == y_val) / length(y_val);
            
            % Store results in table
            results_table(row, :) = [hidden_size, lr, val_accuracy];
            row = row + 1;
            
            % Update best parameters if current combination is better
            if val_accuracy > optimal_accuracy
                optimal_accuracy = val_accuracy;
                optimal_params.hidden_size = hidden_size;
                optimal_params.learning_rate = lr;
                best_net = trial_net;
            end
        end
    end
    
    % Inside the main loop, replace the results display with:
    all_results{currentUserNum} = results_table;  % Store results for current user
    
    % Store grid search results for combined heatmap
    grid_data = reshape(results_table(:,3), [length(learning_rates), length(hidden_layer_sizes)])' * 100;
    all_accuracies(:,:,currentUserNum) = grid_data;

    % Display best parameters
    disp(['Best Hidden Size: ', num2str(optimal_params.hidden_size)]);
    disp(['Best Learning Rate: ', num2str(optimal_params.learning_rate)]);
    disp(['Best Validation Accuracy: ', num2str(optimal_accuracy * 100), '%']);
    
    % Use best network for final predictions
    net = best_net;
    
    % Continue with predictions using the best model
    y_train_raw = net(train_data');
    y_train_pred = double(y_train_raw >= 0.5)';

    % Calculate training accuracy
    train_accuracy = sum(y_train_pred == y_train) / length(y_train);
    disp(['Training Accuracy for Target User ', currentUserID, ': ', num2str(train_accuracy * 100), '%']);

    % Evaluate the trained network on testing data
    y_test_raw = net(test_data');
    y_test_pred = double(y_test_raw >= 0.5)'; % Use same fixed threshold

    % Calculate testing accuracy
    test_accuracy = sum(y_test_pred == y_test) / length(y_test);
    disp(['Test Accuracy for Target User ', currentUserID, ': ', num2str(test_accuracy * 100), '%']);

    % Accumulate total accuracy
    overall_accuracy = overall_accuracy + test_accuracy;

    % Accumulate predictions and true labels for all users
    true_labels = [true_labels; y_test];
    predicted_labels = [predicted_labels; y_test_pred];

end

% Add this after the main loop ends (after "end")
% Display combined results table for all users
fprintf('\nCombined Grid Search Results for All Users:\n');
fprintf('User ID | Hidden Neurons | Learning Rate | Validation Accuracy\n');
fprintf('--------------------------------------------------------\n');
for userNum = 1:10
    userID = sprintf('U%02d', userNum);
    user_results = all_results{userNum};
    for i = 1:size(user_results, 1)
        fprintf('%7s | %13d | %12.3f | %18.2f%%\n', ...
            userID, user_results(i,1), user_results(i,2), user_results(i,3)*100);
    end
end
fprintf('--------------------------------------------------------\n\n');

% Create combined heatmap for all users
figure('Name', 'Combined Grid Search Results - All Users', 'Position', [100 100 800 600]);

% Calculate mean accuracy and standard deviation across all users
mean_accuracies = mean(all_accuracies, 3);
std_accuracies = std(all_accuracies, 0, 3);

% Create heatmap
h = heatmap(learning_rates, hidden_layer_sizes, mean_accuracies);
title('Mean Validation Accuracy (%) Across All Users');
xlabel('Learning Rate');
ylabel('Hidden Neurons');
colormap(jet);

% Customize heatmap appearance
h.FontSize = 12;

% Add text annotations for standard deviation directly in the title
title(sprintf('Mean Validation Accuracy (%%) Across All Users\nStd Dev Range: %.1f%% - %.1f%%', ...
    min(std_accuracies(:)), max(std_accuracies(:))));

% Calculate inter-variance across all users

% Calculate inter-variance across all users
feature_means = mean(combined_user_data, 1);
feature_stds = std(combined_user_data, 0, 1);

% Avoid division by zero by replacing zero means with a small value
feature_means(feature_means == 0) = eps;
inter_variance = mean(feature_stds ./ feature_means);

% Display intra-variance for each user
disp('Intra-variance for each user:');
for userNum = 1:10
    disp(['User ', sprintf('U%02d', userNum), ': ', num2str(user_intra_variance(userNum))]);
end

% Display inter-variance across all users
disp(['Inter-variance across all users: ', num2str(inter_variance)]);

% Calculate and display average accuracy
average_accuracy = overall_accuracy / 10;
disp(['Average Accuracy for All Users: ', num2str(average_accuracy * 100), '%']);

% Plot confusion matrix for training data
figure;
train_targets = zeros(2, length(y_train));
for i = 1:length(y_train)
    train_targets(y_train(i) + 1, i) = 1;
end

outputs_train = zeros(2, length(y_train_pred));
for i = 1:length(y_train_pred)
    outputs_train(y_train_pred(i) + 1, i) = 1;
end
plotconfusion(train_targets, outputs_train);
title('Confusion Matrix for Training Data');

% Plot confusion matrix for testing data
figure;
test_targets = zeros(2, length(true_labels));
for i = 1:length(true_labels)
    test_targets(true_labels(i) + 1, i) = 1;
end

test_outputs = zeros(2, length(predicted_labels));
for i = 1:length(predicted_labels)
    test_outputs(predicted_labels(i) + 1, i) = 1;
end
plotconfusion(test_targets, test_outputs);
title('Confusion Matrix for Testing Data');

% Create figure for user statistics
figure('Position', [100 100 1200 400]);

% Plot mean values
subplot(1,3,1);
user_means = zeros(1,10);
for i = 1:10
    userID = sprintf('U%02d', i);
    user_means(i) = mean(mean(combined_user_data((i-1)*2+1:i*2,:)));
end
bar(user_means);
title('Mean Values per User');
xlabel('User ID');
ylabel('Mean Value');
set(gca, 'XTick', 1:10, 'XTickLabel', {'U01','U02','U03','U04','U05','U06','U07','U08','U09','U10'});

% Plot standard deviations
subplot(1,3,2);
user_stds = zeros(1,10);
for i = 1:10
    userID = sprintf('U%02d', i);
    user_stds(i) = mean(std(combined_user_data((i-1)*2+1:i*2,:)));
end
bar(user_stds);
title('Standard Deviation per User');
xlabel('User ID');
ylabel('Standard Deviation');
set(gca, 'XTick', 1:10, 'XTickLabel', {'U01','U02','U03','U04','U05','U06','U07','U08','U09','U10'});

% Plot intra-variance
subplot(1,3,3);
bar(user_intra_variance);
title('Intra-variance per User');
xlabel('User ID');
ylabel('Intra-variance');
set(gca, 'XTick', 1:10, 'XTickLabel', {'U01','U02','U03','U04','U05','U06','U07','U08','U09','U10'});

% Adjust layout
sgtitle('User Statistics Comparison');

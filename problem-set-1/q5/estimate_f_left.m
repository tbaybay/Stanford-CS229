% %% To prepare the smoothed dataset
% %--------------------------------
% f_train = zeros(450, 200);
% f_test = zeros(450, 50);
%  
% for j = 1:200
%      [~, f_train(:, j)] = lyman_locally_weighted_linear(j, 5, 1);
% end
%  
% for j = 1:50
%      [x, f_test(:, j)] = lyman_locally_weighted_linear(j, 5, 0); 
% end

%% f_left estimator
function [f_left_est, mean_error] = estimate_f_left(f_data)
    [x, ~] = lyman_locally_weighted_linear(1, 5, 1);
    f_left = f_data(x < 1200, :);
    f_right = f_data(x >= 1300, :);
    f_left_est = zeros(size(f_left));
    training_error = zeros(size(f_left, 1), 1);
    k = 3;
    distances = l2_squared(f_right, f_right);

    for j = 1:size(f_right, 2);
        K = knn(distances, j, 3);
        h = max(distances(j, :));
        knn_weights = max([1 - distances(j, K)'./h, zeros(k, 1)], [], 2);
        f_left_est(:, j) = f_left(:, K)*knn_weights./sum(knn_weights);
        training_error(j) = l2_squared(f_left_est(:, j), f_left(:, j));
    end

    mean_error = mean(training_error);
end
%% To prepare the smoothed dataset
%--------------------------------
% f_train = zeros(450, 200);
% f_test = zeros(450, 50);
% 
% for j = 1:200
%     [~, f_train(:, j)] = lyman_locally_weighted_linear(j, 5, 1);
% end
% 
% for j = 1:50
%     [x, f_test(:, j)] = lyman_locally_weighted_linear(j, 5, 0); 
% end

%% f_left estimator
[x, ~] = lyman_locally_weighted_linear(1, 5, 1);
f_left = f_train(x < 1200, :);
f_right = f_train(x >= 1300, :);
f_left_est = [];
k = 3;
distances = l2_squared(f_right, f_right);

for j = 1:size(f_right, 2);
    K = knn(distances, j, 3);
    h = max(distances(j, :));
    numerator = numerator + f_left(j, :)*max([1 - distances(repmat(j, k, 1), K./h), 0]);
    denominator = denominator + max(
    
end
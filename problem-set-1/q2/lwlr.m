function [y, theta] = lwlr(X_train, y_train, x, tau)
% Fits a logistic regression model to training data, then makes predictions
% on x. Tau is convergence tolerance

% Receives (x(i) ? R2) and outputs (y(i) ? {?1, 1}) (Binary classification)
% Implements Newton-Raphson to minimize average empiral loss:
% $J(\theta) = \frac{1/m}\sum_{i = 1}^{m}\log{(1 + \exp{-y^{(i)}\theta^Tx^{(i)})}$

theta = zeros(2, 1);
convergence_tolerance = tau*ones(2, 1);
del_J = [1; 1];
J = 1;
i = 0;
param_ax = subplot(2, 1, 1);
cost_ax = subplot(2, 1, 2);
plot(cost_ax, [-10e5, 10e5], [convergence_tolerance(1); convergence_tolerance(1)], 'k-');
hold(param_ax, 'on'); hold(cost_ax, 'on');

while abs(del_J) > convergence_tolerance
    plot(param_ax, i, theta(1), 'r.'); plot(param_ax, i, theta(2), 'b.');
    plot(cost_ax, i, J, 'k.');
    J = cost(X_train, y_train, theta);
    del_J = cost_gradient(X_train, y_train, theta);
    theta = theta + 0.1*inv(cost_hessian(X_train, y_train, theta))*del_J;
    i = i + 1;
end

y = zeros(length(x), 1);
for i = 1:length(x)
    y(i) = 1./(1 + exp(-theta'*x(:, i)));
end

xlabel(cost_ax, 'Iteration'); ylabel(param_ax, 'Parameter value'); ylabel(cost_ax, 'Cost');
title(param_ax, 'Parameter values and cost vs. iterations'); xlim(param_ax, [0, i]);
xlim(cost_ax, [0, i]);
end

function H = cost_hessian(X_train, y_train, theta)
    m = size(X_train, 2); % # samples
    n = size(X_train, 1); % # features
    H = zeros(n, n);
    for i = 1:m
        exp_term = exp(y_train(i).*theta'*X_train(:, i));
        sample_H = (y_train(i).^2).*X_train(:, i)*X_train(:, i)'.*exp_term./((1 + exp_term).^2);
        H = H + sample_H;
    end
    H = H./m;
end

function del_J = cost_gradient(X_train, y_train, theta)
    m = size(X_train, 2); % # samples
    n = size(X_train, 1); % # features
    del_J = zeros(n, 1);
    for i = 1:m
        sample_del = y_train(i).*X_train(:, i)./(1 + exp(y_train(i).*theta'*X_train(:, i)));
        del_J = del_J + sample_del;
    end
    del_J = del_J./m;
end

function J = cost(X_train, y_train, theta)
    m = size(X_train, 2); % # samples
    J = 0;
    for i = 1:m
        sample_cost = log(1 + exp(-y_train(i).*theta'*X_train(:, i)));
        J = J + sample_cost;
    end
   J = J./m;
end
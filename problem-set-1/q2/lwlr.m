function y = lwlr(X_train, y_train, x, tau)
% Fits a logistic regression model to training data, then makes predictions
% on x. Tau is (?)

% Receives (x(i) ? R2) and outputs (y(i) ? {?1, 1}) (Binary classification)
% Implements Newton-Raphson to minimize average empiral loss:
% $J(\theta) = \frac{1/m}\sum_{i = 1}^{m}\log{(1 + \exp{-y^{(i)}\theta^Tx^{(i)})}$
% Initialize Newton’s method with ? = ~0 (the vector of all zeros). 
% What are the coefficients ?
% resulting from your fit? (Remember to include the intercept term.)

theta = zeros(2, 1);
convergence_tolerance = 1e-2;
error = 10;
i = 0;

while error > convergence_tolerance
    error = cost_gradient(X_train, y_train, theta);
    theta = theta -  cost_hessian(X_train, y_train)\error;
    plot(i, theta(1), 'ro');
    hold on
    plot(i, theta(2), 'bo');
end

end

function H = cost_hessian(X_train, y_train)
    m = size(X_train, 1); % # samples
    n = size(X_train, 2); % # features
    for i = 1:m
        exp_term = exp(y_train(i).*theta'*X_train(i, :));
        sample_H = (y_train(i).^2).*X_train(i, :)*X_train(i, :)'.*exp_term./((1 + exp_term).^2);
        H = H + sample_H;
    end
    H = H./m;
end

function del_J = cost_gradient(X_train, y_train, theta)
    m = size(X_train, 1); % # samples
    n = size(X_train, 2); % # features
    del_J = zeros(n, 1);
    for i = 1:m
        sample_del = y_train(i).*X_train(i, :)./(1 + exp(y_train(i).*theta'*X_train(i, :)));
        del_J = del_J + sample_del;
    end
    del_J = del_J./m;
end
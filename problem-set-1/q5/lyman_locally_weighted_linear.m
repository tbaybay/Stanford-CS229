% A larger tau value resulted in a flatter allocation of weights among
% adjacent points, meaning that the 'local' fit was heavily influenced
% by points that were further away. This resulted in all thetas being
% similar, producing a more-or less constant-gradient result.
% This was in contrast to a small tau value, which meant that nearby points
% were weighted heavily and those further away ignored - this resulted
% in the theta reflecting the gradient in the direct neighborhood more closely.

% J = sum(w(i) * (y(i) - theta*x(i))^2 
% w(i) = exp(-(x - x(i)).^2 ./ 2*tau.^2)

function [query_x, qso_estimate] = lyman_locally_weighted_linear(sample_number, tau, train)
    [lambdas, train_qso, test_qso] = load_quasar_data();
    if train == 1
        qso = train_qso;
    elseif train == 0
        qso = test_qso;
    else
        error('Selected set must be 1 (train) or 0 (test).');
    end
    
    query_x = (min(lambdas):max(lambdas))';
    qso_estimate = zeros(length(query_x), 1);
    y = qso(sample_number, :);

    for i = 1:length(query_x)
        w = exp(-(query_x(i) - lambdas).^2/(2.*(tau.^2)));
        W = diag(w, 0);
        theta = (lambdas'*W*y')./(lambdas'*W*lambdas);
        qso_estimate(i) = theta*query_x(i);
    end
    
    return
end
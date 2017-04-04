[lambdas, train_qso, test_qso] = load_quasar_data();

query_x = (min(lambdas):max(lambdas))';
tau = 5;
qso_estimate = zeros(length(query_x), 1);

for i = 1:length(query_x)
    w = exp(-(query_x(i) - lambdas).^2/(2.*(tau.^2)));
    W = diag(w, 0);
    theta = (lambdas'*W*train_qso(1, :)')./(lambdas'*W*lambdas);
    qso_estimate(i) = theta*query_x(i);
end

plot(lambdas, train_qso(1, :), 'b-');
hold on
plot(query_x, qso_estimate, 'r-', 'LineWidth', 3);
    
% Demo
clf;
[X_train, y_train] = load_data();
% plot3(X_train(:, 1), X_train(:, 2), y_train, 'k.'); axis square;
[y_hat, theta] = lwlr(X_train', y_train', X_train', 10e-20);

figure;
plot3(X_train(y_hat > 0.5, 1), X_train(y_hat > 0.5, 2), y_train(y_hat > 0.5), 'ro');
hold on
plot3(X_train(y_hat < 0.5, 1), X_train(y_hat < 0.5, 2), y_train(y_hat < 0.5), 'bo');
plot3(X_train(:, 1), X_train(:, 2), y_hat, 'k.');
xlabel('X_1'); ylabel('X_2'); zlabel('y');
legend({'$\hat{y} = 1$', '$\hat{y} = 0$'}, 'Interpreter', 'latex')
axis square
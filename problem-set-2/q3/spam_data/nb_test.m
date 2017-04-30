
[spmatrix, tokenlist, category] = readMatrix('MATRIX.TEST');

testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);

% Assume nb_train.m has just been executed, and all the parameters computed/needed
% by your classifier are in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Construct the (numTestDocs x 1) vector 'output' such that the i-th entry 
% of this vector is the predicted class (1/0) for the i-th  email (i-th row 
% in testMatrix) in the test set.
output = zeros(numTestDocs, 1);
p_spam = zeros(numTestDocs, 1);
% Used logarithms to avoid problems with underflow (although it looked like
% Matlab was handling this fine?)
for j = 1:numTestDocs
    % Get the tokens that appear
    tk_in_msg = find(testMatrix(j, :) > 0);
    tk_nin_msg = find(testMatrix(j, :) == 0);
    px_g_spam = exp(sum(log(phi_spam(tk_in_msg))) + sum(log(1 - phi_spam(tk_nin_msg))) + log(phi_y));
    px_g_nonspam = exp(sum(log(phi_nonspam(tk_in_msg))) + sum(log(1 - phi_spam(tk_in_msg))) + log(1 - phi_y));
    p_spam(j) = px_g_spam./(px_g_spam + px_g_nonspam);
    output(j) = (p_spam(j) > 0.5);
end

%---------------
%---------------


% Compute the error on the test set
y = full(category);
y = y(:);
error = sum(y ~= output) / numTestDocs;

%Print out the classification error on the test set
fprintf(1, 'Test error: %1.4f\n', error);




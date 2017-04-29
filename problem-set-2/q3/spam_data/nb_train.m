
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


%% SOLUTION %%

% What are the parameters we're fitting?
% Two parameters for each token:
%   > The probability of that token given y =1
%   > The probability of that token given y = 0
% And phi_y, the base rate : p(y = 1)

% How do we fit them?
% By calculating the fraction of occurences 

% How do we use these parameters to make a prediction?
% We use Bayes' theorem to calculate the probability of each class, given
% the query feature vector, then select which class is more probable.

% Right, so we ned to first calculate phi_y
%       -> Marginal probability that an email is spam
trainCategory = full(trainCategory);
phi_y = sum(trainCategory)/length(trainCategory);

% Now we get the sampling distribution for each token: we'll start with the
% spam.
% We're calculating the probability of a word, conditional on whether it's
% spam or not.
phi_spam = zeros(numTokens, 1);
phi_nonspam = zeros(numTokens, 1);
for i = 1:numTokens
    phi_spam(i) = (sum((trainMatrix(:, i) > 0).*(trainCategory(:) == 1)) + 1)./(sum(trainCategory == 1) + 2);
    phi_nonspam(i) = (sum((trainMatrix(:, i) > 0).*(trainCategory(:) == 0)) + 1)./(sum(trainCategory == 0) + 2);
end

% Note that the + 1 and +2s implement Laplace smoothing - in the absence of
% any previous instances of a token, we want the classifier to assign equal
% weights to spam/non-spam.
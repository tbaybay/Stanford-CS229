%% Problem Set 2, Question 3 %%
% Using this script requires that
% nb_train is run first.

% This isn't really log-odds, as we're taking the ratio of 
% p(xi | y) and p(xi | ¬y) as opposed to p(xi) and p(¬xi).

% (b) Which tokens had the largest log(p(xi | y = 1)/p(xi | y = 0))?
logit = log(phi_spam./phi_nonspam);
wordvec = strsplit(tokenlist, ' ');
n_words = 10;
spam_ix = zeros(n_words, 1);
j = 1;
while j < n_words
    ix = find(max(logit) == logit);
    spam_ix(j:(j + length(ix) - 1)) = ix;
    logit(ix) = [];
    j = j + length(ix);
end

% Plot the words with the top 10 logits
bar(log(phi_spam(spam_ix)./phi_nonspam(spam_ix)));
set(gca, 'xticklabel', wordvec(spam_ix), 'xticklabelrotation', 90)
function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%hypothesis(image) = sigmoid(all_theta(1,1) * X(1,1) + alltheta(1, 2) * X(1, 2) + ... + alltheta(1, 400) * X(1, 400))
% sequential implementation: 
% for i = 1:m
%   for j = 1:num_labels
%     p(i) = max(
%       sigmoid(
%         sum(
%           all_theta(i, j) * X(i, j); 
%           )
%         )
%     )
%   endfor % check P(y=j|x;theta)
% endfor % every image	    

% vectorized implementation:
initial_probabilities = zeros(size(X, 1), num_labels);
for c = 1:num_labels
  initial_probabilities(:, c) = sigmoid(X*all_theta(c, :)');
endfor
[x, ix] = max(initial_probabilities, [], 2);
p = ix;

% =========================================================================

end

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
Z = all_theta * X';% since each row of theta is the trained parameters for each number 
%(0-9) in this example (the type of label) and X contains each example we want to predict 
%in rows we need to take all_theta * X transpose in order to get the vectorized
%implementation that way each row of theta will be multiplied by each example in X'
%(columns in X' are now the examples to be predicted)

results = sigmoid(Z); %sigmoid is our function we use to predict the probability
%that a specific example is one of the labels in our training set

[value,p] = max(results', [], 2);
%the max function returns [value at max, index of max] by taking the transpose of the
%results matrix we got a exampleSetSize x numberOfLabelsSize matrix (e.g. 5000 x 10 
%in our case). Therefore each row corresponded to each row in the original
%dataset and each column corresponded to the probability it was one of those labels (0-9)
%so taking the max of each row, and getting the index (the indices correspond to the 
%digits) of the max gave us the predicted digit


% =========================================================================


end

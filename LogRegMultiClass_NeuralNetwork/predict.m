function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%USING size to check matrix dimensions when debugging

% Add column of ones to the X data matrix
X = [ones(m, 1) X];
%size(X)

z2 = X * Theta1'; %calculating z2 to be #ofExamples by #ofTheta1Rows 
%There are 25 units in the second layer of our neural network so we want a n x 25 vector
% n is the number of examples here

%size(z2)
a2 = sigmoid(z2); %calculate sigmoid on first layer and first theta matrix

%size(a2)

%Add column of ones to a2 to get n x 26 matrix
a2 = [ones(size(a2,1),1) a2]; 

z3 = a2 * Theta2'; %want a resulting n x 10 matrix where rows are examples
%and columns are probabilities that the example fits within that classification
a3 = sigmoid(z3);

[value,p] = max(a3, [], 2);

%the max function returns [value at max, index of max] by taking the transpose of the
%results matrix we got a exampleSetSize x numberOfClassificationsSize matrix 
%(e.g. 5000 x 10 in our case). Therefore each row corresponded to each row in the original
%dataset and each column corresponded to the probability it was one of those labels (0-9)
%so taking the max of each row, and getting the index (the indices correspond to the 
%digits) of the max gave us the predicted digit



% =========================================================================


end

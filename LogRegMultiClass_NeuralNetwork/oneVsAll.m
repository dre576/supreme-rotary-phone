function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
     % Set Initial theta
     %initial_theta = zeros(n + 1, 1);
%     
     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
     % Run fmincg to obtain the optimal theta
     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);

for i = 1:num_labels
%this for loop is designed to loop through to train a number of different logistic 
%regression parameters (one for each digit we want to learn in this example)
%the digits go from 0-9, but they mapped the 0 digit to correspond to 10 in the y vector
%because octave doesn't start it's indexing at 0, instead it starts at 1


% Set Initial theta column vector of size n+1
     initial_theta = zeros(n + 1, 1);
     
% Set options for fminunc
     options = optimset('GradObj', 'on', 'MaxIter', 50);
 
% Run fmincg to obtain the optimal theta
% This function will return theta and the cost 
     [theta] = ...
         fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), ...
                 initial_theta, options);
%since y is a vector of labels ranging from 1 to the number of labels we have to use the 
%i variable for the logical array comparison

%by doing y == i this is actually returning a vector of size(y) that is getting passed 
%into fmincg of only 1s and 0s to be used in the one versus all training
%e.g. say vector y = [1;1;1;2;2;2;3;3;3] (size of 9x1) then let this be the first 
%iteration so i = 1, the return vector r would be r = [1;1;1;0;0;0;0;0;0]
%second iteration i = 2, the return vector r = [0;0;0;1;1;1;0;0;0] etc.
%this vector tells the learning algorithm which type it's training on, which one
% is a positive hit (a 1) or a negative (not part of the set is a 0)



all_theta(i,:) = theta';
%since each row of all_theta needs to correspond to the theta values calculated for each 
%learned logistic regression run we have that each row of all_theta is equal to the 
%transpose of the calculated column vector theta
end



% =========================================================================


end

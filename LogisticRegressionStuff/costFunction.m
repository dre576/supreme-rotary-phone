function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
temp1 = log(sigmoid(X*theta));
temp2 = log(1-sigmoid(X*theta));
cost1 = y'*temp1; %this is y transpose times the temp1 matrix in order to take
%advantage of matrix operations (y transpose * x is the element wise sum of 
%the two vectors)
cost2 = (1-y)'*temp2;
J = (-1/m)*(cost1 + cost2);

temp3 = sigmoid(X*theta) - y; %this is h_theta(x) where x is theta transpose 
%times a row of the X matrix (one training example is a row of this matrix): 
%so here we use matrix X times theta (sigmoid is the h_theta() function)
%to achieve that and send it in to the sigmoid function, then subtract y to get 
%the difference in prediction and actual value (result is a n by 1 vector
% because it's just a scalar number computed for each row)

d = (X'*temp3)./m; %since this is the gradient (uses the derivative) We want 
%the sum of all rows of X times their respective scalars 
%example: say X is a 100x3 then temp3 would be 100x1 doing X transpose * temps 3
% gives us a 3x1 vector which is what we would want because we have a theta that
% is 3x1 as well!

grad = d;






% =============================================================

end

function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

thetaSum = theta(2:end,1)'*theta(2:end,1); %want the sum of all the thetas squared leaving 
%out the first theta (theta_0)
lambdaPart = lambda/(2*m) * thetaSum;

J = (1/(2*m))*sum((X*theta - y).^2) + lambdaPart;
%linear regression is just the sum of the thetas times their respective features(columns in the X
% matrix for each example from the training set

grad = (X'*((X*theta) - y))/m;
%Update everything as if it was a regular gradient function

grad(2:end) = grad(2:end) + ((lambda/m)*theta(2:end));
%Then update everything other than the first grad entry as it should not include 
%the regularization term with lambda








% =========================================================================

grad = grad(:);

end

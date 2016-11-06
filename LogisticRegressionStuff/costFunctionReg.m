function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
temp1 = log(sigmoid(X*theta));
temp2 = log(1-sigmoid(X*theta));
cost1 = y'*temp1; %this is y transpose times the temp1 matrix in order to take
%advantage of matrix operations (y transpose * x is the element wise sum of 
%the two vectors)
cost2 = (1-y)'*temp2;
thetaSum = theta(2:end,1)'*theta(2:end,1); %want the sum of all the thetas squared leaving 
%out the first theta (theta_0)
lambdaPart = lambda/(2*m) * thetaSum;

J = (-1/m)*(cost1 + cost2)+lambdaPart;

grad = (X'*(sigmoid(X*theta) - y))/m;
%Update everything as if it was a regular gradient function

grad(2:end) = grad(2:end) + ((lambda/m)*theta(2:end));
%Then update everything other than the first grad entry as it should not include 
%the regularization term with lambda





% =============================================================

end

function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%



mu = sum(X,1)'./m; %summing each column (feature) over all the examples then dividing by 
%the number of examples to get the mean (took transpose to get it as a column vector

temp = X - mu'; %since we took the transpose earlier and the examples are in rows with features 
%as the column we need to transpose mu back for use here

sigma2 = sum(temp.^2,1)'./m; %summing over all of the examples to get sigma^2 (variance) and dividing
%by number of examples


% =============================================================


end

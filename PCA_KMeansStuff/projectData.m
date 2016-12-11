function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%
U_reduce = U(:,1:K); %get the top K vectors of U
Z = (X * U_reduce); %U_reduce is nxK, X is mxn, therefore we need to project X onto
%U_reduce to take advantage of vectorization (X is mxn because its examples are the rows,
%U_reduce is nxK because its eigenvectors are its columns)
%Z is mxK



% =============================================================

end

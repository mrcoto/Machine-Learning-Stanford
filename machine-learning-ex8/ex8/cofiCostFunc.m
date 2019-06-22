function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


% J = (1/2) * sum((i, j):r(i, j) = 1 de [theta~(j)'*x~(i) - y~(i,j)]^2)
J = (1/2) .* sum(sum(((X*Theta' - Y) .* R).^2));

% grad_x = sum(j:r(i,j) = 1 de [theta~j' * x~i - y~(i,j)]*theta_k~j)
X_grad = ((X*Theta' - Y) .* R) * Theta;

% grad_t = sum(j:r(i,j) = 1 de [theta~j' * x~i - y~(i,j)]*x_k~j)
Theta_grad = ((X*Theta' - Y) .* R)' * X;

% Regularizando función de costo J
% Jreg = (1/2)(...) + (lambda/2) * sum(de j = 1 a n_u de sum(de k = 1 a n de [theta_k~j]^2)) + (lambda/2) * sum(de j = 1 a n_m de sum(de k = 1 a n de [x_k~j]^2))
J = J + (lambda/2) * sum(sum(Theta .^2)) + (lambda/2) * sum(sum(X .^2));

% Regularizando gradiente
%grad_x = sum(j:r(i,j) = 1 de ...) + lambda * x_k~i
X_grad = X_grad + lambda * X;

%grad_t = sum(j:r(i,j) = 1 de ...) + lambda * theta_k~i
Theta_grad = Theta_grad + lambda * Theta;



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end

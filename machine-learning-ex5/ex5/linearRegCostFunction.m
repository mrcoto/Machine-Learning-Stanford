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

% NOTA: No se regulariza theta para j = 1, o en octave: theta(1)

% J = (1/2m) * sum(de i a m de (h - y)^2) + (lambda/m) * sum(de 1 a n de theta^2)  

theta_ = theta; 
theta_(1) = 0; % Setea el primer elemento como 0 (no se regulariza el primero)
h = X * theta;
J = (1/(2*m)) * sum( (h - y).^2 ) + (lambda/(2*m)) * sum(theta_.^2);


% grad = (1/m) * sum(de 1 a m de (h - y)*x)  para j = 0
% grad = ( (1/m) * sum(de 1 a m de (h - y)*x) ) + (lambda/m) * theta  para j > 0

% h - y --> 12x1    x --> 12x2      theta_ --> 2x1
grad = (1/m) .* sum( (h - y) .* X ) + (lambda/m) .* theta_';




% =========================================================================

grad = grad(:);

end

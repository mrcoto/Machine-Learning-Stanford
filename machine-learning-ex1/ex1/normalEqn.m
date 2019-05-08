function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------
% X -> 47x3     y -> 47x1   theta -> 3x1
% X' * X --> 3x47 * 47x3 --> 3x3
% pinv(3x3) --> 3x3
% pinv(3x3) * X' --> 3x3 * 3x47 --> 3x47
% ans(3x47)*y --> 3x47 * 47x1 --> 3x1
theta = pinv(X'*X)*X'*y;


% -------------------------------------------------------------


% ============================================================

end

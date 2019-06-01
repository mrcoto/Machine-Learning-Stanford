function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

power = 1:1:p; % si p es 8 genera: [1 2 3 4 5 6 7 8] --> 1x8
power = repmat(power, size(X, 1), 1); % Repite la fila M veces hacia abajo, queda de Mx8

X_poly = repmat(X, 1, p); % si p es 8, repite la columna 8 veces (asume que X es Mx1)  -> Nx8
X_poly = X_poly .^ power;




% =========================================================================

end

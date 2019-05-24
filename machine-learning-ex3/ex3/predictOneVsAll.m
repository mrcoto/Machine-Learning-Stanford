function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% num_labels: 10, p: 5000x1
% X: 5000x401 , all_theta: 10 x 401

% Cada fila en X se multiplica por cada fila en all_theta.
% De esa multiplicación se escoge el mayor valor, el cual corresponde a max h^i(x)
% si el máximo se encontró en la columna '3', entonces la etiqueta corresponde para el dígito '3'

% X * all_theta' ==> 5000 x 10 ==> se escoge el max de las 10 columnas.
% el 2 indica que se buscará el mayor de la fila, lo traduce a una matriz de 5000 x 1
[maximums, labels] = max( X * all_theta', [], 2);  
p = labels;


% =========================================================================


end
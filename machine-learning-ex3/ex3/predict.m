function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1: 25x401    Theta: 10x26      X: 5000x400

% Se añade el bias (la columna de '1') al inicio de X para que quede de 5000x401
X = [ones(m, 1), X];

% Capa 2 (Oculta)
z2 = X * Theta1';     % 5000x25
a2 = sigmoid(z2);     % 5000x25

% Se añade el bias (la columna de '1') al inicio de a2 para que quede de 5000x26
a2 = [ones(m, 1), a2];

% Capa 3 (Salida)
z3 = a2 * Theta2';    % 5000x10
h = sigmoid(z3);      % 5000x10

% Predicción
[maximums, labels] = max(h, [], 2);  % 2: por fila
p = labels;


% =========================================================================


end

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% ========= [ex4 Part 3: Compute Cost (Feedforward)] =============

% cada y(i) contiene un número del 1 al 10, se debe recodear. Ej: 5 ---> [0 0 0 0 1 0 0 0 0 0]
I = eye(num_labels); % Matriz identidad num_labels x num_labels, I(5, :) retorna [0 0 0 0 1 0 0 0 0 0]
Y = zeros(m, num_labels); % 5000x10
for i=1:m
    digit = y(i);
    Y(i, :) = I(digit, :);
endfor;

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

% Y: 5000x10
% sum(1 a m de sum(1 a K de [-y log (h) - (1 - y) log(1 - h)]))
J = sum( sum( -Y .* log(h) - (1 - Y) .* log(1 - h) ) ) / m;

% ========= [ex4 Part 4: Implement Regularization] =============
% J = J + (lambda/2m) * [ sum(1 a 25 de sum(1 a 400 de Theta1^2) + sum(1 a 10 de sum(1 a 25 de Theta2^2))) ]
% Theta1: 25x401    Theta2: 10x26

% Se remueve la primera columna asociado al bias de Theta1 y Theta2.
Theta1_ = Theta1(:, [2:end]);    % Theta1_: 25x400
Theta2_ = Theta2(:, [2:end]);    % Theta2_: 10x25

% Se añade el término de regularización a la función de costo
J = J + (lambda/(2*m)) * ( sum(sum(Theta1_.^2)) + sum(sum(Theta2_.^2)) );

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%   Theta1: 25x401      Theta2: 10x26   z3: 5000x10

% Error capa 3
error_3 = h - Y;  % 5000x10;

% Error capa 2  (theta(l))T * error_(l+1) .* g'(z(l))   l: capa
z2_ = [ones(m, 1), z2];  % Se añade el bias
error_2 = (error_3 * Theta2) .* sigmoidGradient(z2_);
error_2 = error_2(:, 2:end);    % Mantiene todas las filas, y remueve la primera columna

% deltas  (delta(l) = delta(l) + a(l) * error(l+1))     l: capa
% delta_inicial es 0
delta_1 = error_2' * X; 
delta_2 = error_3' * a2;

% Derivada de J (O Gradiente)
Theta1_grad = delta_1 ./ m;
Theta2_grad = delta_2 ./ m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% D o Theta_grad. 
% D = D para j = 0
% D = D + (lambda/m) * Theta(l) ára j != 0      l: capa

% Nota: En este punto, Theta1/2 ya tiene el bias en la primera columna.
% Como la regularización suma un término con valor '0' cuando j es 0, se reemplazará la primera 
% columna de unos por una columna de ceros 
Theta1_ = [ zeros(size(Theta1, 1), 1), Theta1(:, 2:end) ];
Theta2_ = [ zeros(size(Theta2, 1), 1), Theta2(:, 2:end) ];

Theta1_grad = Theta1_grad + (lambda/m) .* Theta1_;
Theta2_grad = Theta2_grad + (lambda/m) .* Theta2_;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

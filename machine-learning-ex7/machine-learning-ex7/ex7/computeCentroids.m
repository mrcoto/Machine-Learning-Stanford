function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

% u_k = 1/|Ck| sum(i en Ck de x(i))     <-- Ck: set de ejemplos
% Ej, si k=2, se busca ejemplos asignados a Centroide 2: C2
% Si se asume que x(3) y x(5) están ahí, entonces: |C2| = 2 y sum() es de x(3) + x(5)
for k=1:K
    example_set = idx == k; % Retorna 1 para los que son iguales y 0 para los que no
    Ck = sum(example_set);
    Xset = X .* example_set; % Solo deja los que tienen '1' asignado (ya que 0 * algo es 0)
    centroids(k, :) = (1/Ck) .* sum(Xset, 1)
endfor



% =============================================================


end


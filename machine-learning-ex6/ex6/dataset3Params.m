function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Datos a probar:
data = [0.01 0.03 0.1 0.3 1 3 10 30];
len = size(data, 2);
min_error = 999999;

% Loop de todas las combinaciones data X data
for ci=1:len
    for sj=1:len
        model = svmTrain(X, y, data(ci), @(x1, x2) gaussianKernel(x1, x2, data(sj)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        % Establece C y sigma para el menor error encontrado
        if (error < min_error) 
            C = data(ci);
            sigma = data(sj);
            min_error = error;
        endif;
    endfor;
endfor;

printf("C=%d, sigma=%d\n", C, sigma);

% =========================================================================

end

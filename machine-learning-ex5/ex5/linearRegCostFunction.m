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
% size(grad)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

pred = X * theta;
sqrErrors = (pred - y).^2;
regularization_term = lambda * sum(theta(2:end) .^ 2)/(2*m);
J = 1 / (2*m) * sum(sqrErrors) + regularization_term;

% X
grad = (X' * (pred - y))/m;
temp = theta;
grad(2:end) = grad(2:end) .+ (lambda/m) .* temp(2:end);
% grad(2:n) = grad(2:n) + lambda / m * theta(2:n);


% htheta = X * theta;
% n = size(theta);
% J = 1 / (2 * m) * sum((htheta - y) .^ 2) + lambda / (2 * m) * sum(theta(2:n) .^ 2);

% grad = 1 / m * X' * (htheta - y);
% size(grad)
% grad(2:n) = grad(2:n) + lambda / m * theta(2:n);










% =========================================================================

grad = grad(:);

end

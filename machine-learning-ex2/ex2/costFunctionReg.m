function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

pred = sigmoid(X * theta);
err = -y .* log(pred) - (1 - y) .* log(1 - pred);
cost_ = sum(err)/m;
reg_ = sum(theta(2:end) .^ 2) .* (lambda/(2*m));
J = cost_ + reg_ ;

grad0 = sum((pred - y) .* X(:,1))/m;

% printf('grad0\n');
% disp(size(grad0));
% printf('second\n');
% disp(size(sum((pred - y) .* X(:,1:2))));
% printf('third\n');
% disp(size(lambda .* theta(2:length(theta))'))
% temp = sum((pred - y) .* X(:,1:2))/m + (lambda .* theta(2:end))'/m;
% temp = (sum((pred -y) .* X(:,1:2)))/m + (lambda .* theta(2:end))'/m;
% grad = [grad0 temp]
grad(1) = 1 / m * sum((pred - y) .* X(:, 1));
for i = 2:size(theta, 1)
    grad(i) = 1 / m * sum((pred - y) .* X(:, i)) + lambda / m * theta(i);
end



% =============================================================

end

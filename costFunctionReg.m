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



z = X * theta;
h = sigmoid(z);
cost = (-y .* log(h))-(1-y).*log(1-h);
term1 = (1/m) * sum(cost);
theta_mod = theta([2:size(theta,1)],:); %skip theta_0 (or in octave, theta_1)
term2 = (lambda/(2*m))*sum( theta_mod.^2  );
J = term1 + term2; 

% get the "normal" unregularized grad
grad =(1/m) * ((h-y)' * X)';
% other than the first theta term, everything else addes a reg term
regTerm = (lambda/m) * theta;
% the first theta term will remain the same.  save it first.
grad1 = grad(1);
% the rest of the terms need to add reg term.  we simply do it for every term for now
grad = grad + regTerm;
% then we substitute back the first term from where we saved it.
grad(1) = grad1;


% =============================================================

end

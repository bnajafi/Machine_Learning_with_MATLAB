function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

theta_remain=theta(2:end);
Z= (theta'*X')';
h_theta = sigmoid(Z);

J1 = -y.*log(h_theta);
J2=-(1*ones(m, 1)-y).*log(1*ones(m, 1)-h_theta);
J=sum(J1+J2)/m+(lambda/(2*m))*sum(theta_remain.^2);


% grad_firstPart = (1/m)*(h_theta-y)'*X;
% grad_regPart = ((lambda/(2*m))*[0;theta_remain])';
% grad = grad_firstPart+grad_regPart;
grad_previous= (1/m)*(h_theta-y)'*X;
grad0=grad_previous(1);
grad_remain_first=grad_previous(2:end);
grad_remain_second=((lambda/(m))*theta_remain)';
grad_remain=grad_remain_first+grad_remain_second;
grad=[grad0,grad_remain];



end

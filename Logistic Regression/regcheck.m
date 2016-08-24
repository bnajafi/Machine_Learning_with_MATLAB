
clear ; close all; clc
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
plotData(X, y);
hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;



X = mapFeature(X(:,1), X(:,2));

initial_theta = zeros(size(X, 2), 1);
m = length(y)
lambda = 1;
theta=initial_theta
theta_remain=theta(2:end);
Z= (theta'*X')';
h_theta = sigmoid(Z);

J1 = -y.*log(h_theta);
J2=-(1*ones(m, 1)-y).*log(1*ones(m, 1)-h_theta);
J=sum(J1+J2)/m+(lambda/(2*m))*sum(theta_remain.^2);

grad_previous= (1/m)*(h_theta-y)'*X;
grad0=grad_previous(1);
grad_remain_first=grad_previous(2:end);
grad_remain_second=((lambda/m)*theta_remain)';
grad_remain=grad_remain_first+grad_remain_second;
grad=[grad0,grad_remain];
% grad_firstPart = (1/m)*(h_theta-y)'*X;
% grad_regPart = ((lambda/(2*m))*[0;theta_remain])';
% grad = grad_firstPart+grad_regPart;
% 

% Compute and display initial cost and gradient for regularized logistic
% regression
% [cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% %% ============= Part 2: Regularization and Accuracies =============
% %  Optional Exercise:
% %  In this part, you will get to try different values of lambda and 
% %  see how regularization affects the decision coundart
% %
% %  Try the following values of lambda (0, 1, 10, 100).
% %
% %  How does the decision boundary change when you vary lambda? How does
% %  the training set accuracy vary?
% %
% 
% % Initialize fitting parameters
% initial_theta = zeros(size(X, 2), 1);
% 
% % Set regularization parameter lambda to 1 (you should vary this)
% lambda = 1;
% 
% % Set Options
% options = optimset('GradObj', 'on', 'MaxIter', 400);
% 
% % Optimize
% [theta, J, exit_flag] = ...
% 	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);
% 
% % Plot Boundary
% plotDecisionBoundary(theta, X, y);
% hold on;
% title(sprintf('lambda = %g', lambda))
% 
% % Labels and Legend
% xlabel('Microchip Test 1')
% ylabel('Microchip Test 2')
% 
% legend('y = 1', 'y = 0', 'Decision boundary')
% hold off;
% 
% % Compute accuracy on our training set
% p = predict(theta, X);
% 
% fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
% 

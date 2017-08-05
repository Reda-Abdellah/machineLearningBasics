function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)



m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


numbOfFueatures=size(X)(1,2);
for iter = 1:num_iters
thtemp=theta;

for i=1:numbOfFueatures
thtemp(i,1)=thtemp(i,1)-alpha/m*(X*theta-y)'*X(:,i);
end

theta=thtemp;
   
    J_history(iter) = computeCostMulti(X, y, theta);

end

end






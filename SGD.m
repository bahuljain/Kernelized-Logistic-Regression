function [W, iterations, costT] = SGD(Y, K, YT, KT, eta)
%StochasticGradientDescent This function finds the optimum model by
%incrementally updating the model using only one random point.

% Y - Vector of desired output values in training set 
% K - gram matrix for training input values
% YT - Vector of desired output values in test set
% KT - gram matrix for test input values
% eta - step size
% costT - test cost iteration by iteration
% W - model after BGD converges
% iterations - number of iterations in which BGD converges

n = length(Y);

W = ones(1000,1);

epsilon = 1e-2;
lambda = 1e-3;
iterations = 0;
sigma = Sigma(Y .* (W'*K)');
i = randi(1000);
delta = (-K(:,i)*Y(i) + K(:,i)*(Y(i)*sigma(i))) + 2*lambda*W;

tic
while norm(delta) > epsilon && toc < 10
    iterations = iterations + 1
    costT(iterations) = CostFunction(W, KT, YT);
    cost = costT(iterations)
    %disp(strcat(num2str(iterations), ' : ' , num2str(error)))
    i = randi(1000);
    delta = (-K(:,i)*Y(i) + K(:,i)*(Y(i)*sigma(i))) + 2*lambda*W;
    W = W - delta*eta;
    norm(delta)

end
toc
end


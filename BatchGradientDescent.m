function [W, iterations, costT] = BatchGradientDescent(Y,K,YT,KT,eta)
%% BatchGradientDescent This function finds the optimum model by incrementally updating the model.

% Y - Vector of desired output values in training set 
% K - gram matrix for training input values
% YT - Vector of desired output values in test set
% KT - gram matrix for test input values
% eta - step size
% costT - test cost iteration by iteration
% W - model after BGD converges
% iterations - number of iterations in which BGD converges

n = length(Y);

%% Assigning all ones to the model in the beginning
W = ones(n,1);

%% Setting parameter values (eta given by user)
epsilon = 1e-2;
lambda = 1e-3;

%% First update
delta = (K*Y - K*(Y .* Sigma(Y .* (W'*K)')))/(-n)+ 2*lambda*W;
iterations = 0;


%% Iteratively updating the model using stochastic gradient descent
tic
while norm(delta) > epsilon && toc<10
    iterations = iterations + 1;
    costT(iterations) = CostFunction(W, KT, YT);
    delta = (K*Y - K*(Y .* Sigma(Y .* (W'*K)')))/(-n)+ 2*lambda*W;
    gradient = norm(delta)
    W = W - delta*eta;
end
toc
end


function [W, iterations, costT] = SGD100(Y, K, YT, KT, eta)
%% StochasticGradientDescent100 This function finds the optimum model by incrementally updating the model using only 100 random points.

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

%% Setting parameter values and initializing counters (eta given by user)
lambda = 1e-3;
iterations = 0;
minCost = 1e+11;

%% Iteratively updating the model using stochastic gradient descent with 100 random points
tic
while toc < 10
    iterations = iterations + 1;
    
    %% Find cost and use the model corresponding to the minimum cost
    costT(iterations) = CostFunction(W, KT, YT);
    if costT(iterations) < minCost
        minCost = costT(iterations)
        minModel = W; 
    end
    sigma = Sigma(Y .* (W'*K)');

    %% Randomly choosing 100 points
    seq = randperm(n);
    tmpK = K(seq(1:100),:);
    tmpY = Y(seq(1:100));
    tmpsigma = sigma(seq(1:100));
    
    %% Computing gradient and updating model
    delta = (tmpK'*tmpY - tmpK'*(tmpY .* tmpsigma))/(-100)+ 2*lambda*W;
    W = W - delta*eta;    
end
toc
%% Returning model corresponding to the lowest cost
W = minModel;
end


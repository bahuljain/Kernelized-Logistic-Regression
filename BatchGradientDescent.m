function [W, iterations, costT] = BatchGradientDescent(Y,K,YT,KT,eta)
%BatchGradientDescent This function finds the optimum model by
%incrementally updating the model.

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
delta = (K*Y - K*(Y .* Sigma(Y .* (W'*K)')))/(-n)+ 2*lambda*W;

iterations = 0;
tic
while norm(delta) > epsilon% && toc<20% && iterations < 50000
    iterations = iterations + 1
    %costt(iterations) = CostFunction(W, K, Y);
    costT(iterations) = CostFunction(W, KT, YT);
    cost = costT(iterations);
    %disp(strcat(num2str(iterations), ' Gradient: ',num2str(delta),' CostT: ',num2str(costT(iterations))));
    delta = (K*Y - K*(Y .* Sigma(Y .* (W'*K)')))/(-n)+ 2*lambda*W;
    gradient = norm(delta)
    
    W = W - delta*eta;
    
end
toc


end


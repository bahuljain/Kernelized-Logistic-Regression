function [W, iterations, costT] = BatchGradientDescent(Y,K,YT,KT,eta)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
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


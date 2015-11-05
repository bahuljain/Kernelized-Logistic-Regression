function [W, iterations, costT] = SGD(Y, K, YT, KT, eta)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

%W = -1 + (2).*rand(1000,1);
n = length(Y);

W = ones(1000,1);
newW = zeros(1000,1);

epsilon = 1e-5;
lambda = 1e-3;
error = norm(newW - W);
iterations = 0;

tic
while error > epsilon && toc < 10
    iterations = iterations + 1;
    costT(iterations) = CostFunction(W, KT, YT);
    disp(strcat(num2str(iterations), ' : ' , num2str(error)))
    i = randi(1000);
    sigma = Sigma(Y .* (W'*K)');
    delta = (-K(:,i)*Y(i) + K(:,i)*(Y(i)*sigma(i))) + 2*lambda*W;
    newW = W - delta*eta;
    error = norm(newW - W);
    W = newW;
end
toc
end


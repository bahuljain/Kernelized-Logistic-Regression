function [K] = Gram(X1, X2, kappa)
%% Gram Matrix
% X1 - Matrix of input training/testing points
% X2 - Matrix of input TRAINING points only 
% kappa - kappa value
% K - gram matrix

n = length(X1);
K = zeros(n);
for i = 1:n
    for j = 1:n
        K(i,j) = exp(-1*norm(X1(i,:) - X2(j,:))^2 / kappa);
    end
end
end


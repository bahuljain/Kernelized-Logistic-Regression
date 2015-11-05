function [K] = Gram(X1, X2, kappa)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
n = length(X1);
K = zeros(n);
for i = 1:n
    for j = 1:n
        K(i,j) = exp(-1*norm(X1(i,:) - X2(j,:))^2 / kappa);
    end
end


end


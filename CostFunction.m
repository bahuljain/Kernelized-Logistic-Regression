function [cost] = CostFunction( W, K, Y )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
cost = - sum(log(Sigma(Y .* (K*W))));
end


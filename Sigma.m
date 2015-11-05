function [sigma] = Sigma(V)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here

sigma = 1 ./ (1 + exp((-1) * V));    

end


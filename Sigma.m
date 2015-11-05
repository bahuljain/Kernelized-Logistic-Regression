function [sigma] = Sigma(V)
%% Sigmoid Function
% V - input vector
% sigma - output vector after sigmoid function has been applied to every
% element of the vector.

sigma = 1 ./ (1 + exp((-1) * V));    

end


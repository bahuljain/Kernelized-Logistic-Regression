function [prediction, misclassified] = Prediction(W, Y, gram)
% Prediction - Given the optimum model, gram matrix and the desired output
% values the predicted output values are generated and number of
% missclassifications are returned.
%
% W - model
% Y - vector of desired output values
% gram - matrix of input values after applying kernel function
% prediction - vector of predicted output values
% misclassified - number of misclassifications
%

prediction = Sigma(gram*W);
prediction(prediction <= 0.5)= -1;
prediction(prediction > 0.5)= 1;

t = prediction .* Y;
misclassified = length(t(t == -1));
end


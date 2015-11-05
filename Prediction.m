function [prediction, misclassified] = Prediction(W, Y, gram)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
%prediction = sign(W' * gram)';

prediction = Sigma(gram*W);
prediction(prediction <= 0.5)= -1;
prediction(prediction > 0.5)= 1;

t = prediction .* Y;
misclassified = length(t(t == -1));


end


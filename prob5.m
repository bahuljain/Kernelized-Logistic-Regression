%% Driver Program for Kernelized Logistic Regression
clear
clc

%% Loading Data 
% I have already mapped the input points to Hilbert Space using kernel
% function. The code for that can be seen in Gram.m and Kappa.m
load('workspace.mat');

%% Batch Gradient Descent 
[W, iterations, costT] = BatchGradientDescent(TrainingY, gram, TestY, gramTest, 1e-2);
i = 1:iterations;
plot(i./iterations.*10, costT, 'r');
xlabel('time(seconds)');
ylabel('Test Cost');
title('Test Cost vs Time for BGD, SGD(1 point), SGD(100 points)');
hold on

%% Stochastic Gradient Descent using 1 random point
[W, iterations, costT] = SGD(TrainingY, gram, TestY, gramTest, 1e-2);
i = 1:iterations;
plot(i./iterations.*10, costT, 'b');
% title('Test Cost vs Time for Stochastic Gradient Descent(1 points)');
% xlabel('time(seconds)');
% ylabel('Test Cost');

%% Stochastic Gradient Descent using 100 random points
[W, iterations, costT] = SGD100(TrainingY, gram, TestY, gramTest, 1e-2);
i = 1:iterations;
plot(i./iterations.*10, costT, 'g');
legend('Batch','Stochastic (1 point)', 'Stochastic(100 points)');
% title('Test Cost vs Time for Stochastic Gradient Descent(100 points)');
% xlabel('time(seconds)');
% ylabel('Test Cost');

%% Predictions and Misclassifications
% [~,missTrain] = Prediction(W, TrainingY, gram);
% [~,missTest] = Prediction(W, TestY, gramTest);

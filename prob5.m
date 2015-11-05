clear
clc
load('workspace.mat');

% bgd - 1e-2
% sgd - 1e-2
% sgd100 - 1e-2

% [W, iterations, costT] = BatchGradientDescent(TrainingY, gram, TestY, gramTest, 1e-2);
% i = 1:iterations;
% plot(i./iterations.*10, costT, 'r');
% title('Test Cost vs Time for BGD, SGD(1 point), SGD(100 points)');
% xlabel('time(seconds)');
% ylabel('Test Cost');
% hold on
% 
% [W, iterations, costT] = SGD(TrainingY, gram, TestY, gramTest, 1e-2);
% i = 1:iterations;
% plot(i./iterations.*10, costT, 'b');
% %title('Test Cost vs Time for Stochastic Gradient Descent(100 points)');
% 
% [W, iterations, costT] = SGD100(TrainingY, gram, TestY, gramTest, 1e-2);
% i = 1:iterations;
% plot(i./iterations.*10, costT, 'g');
% legend('Batch','Stochastic (1 point)', 'Stochastic(100 points)');
%title('Test Cost vs Time for Stochastic Gradient Descent(100 points)');

[~,missTrain] = Prediction(W, TrainingY, gram);
[~,missTest] = Prediction(W, TestY, gramTest);

% tic
% [W2, iterations2] = SGD(TrainingY, gram, 1e-5);
% toc
%W2 = -1 + (2).*rand(1000,1);
%[prediction2, miss2] = Prediction(W2, TrainingY, gram);    
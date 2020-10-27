clc
clear

%% Import data
X = xlsread('trn_tst\X.csv');
Y = xlsread('trn_tst\Y.csv');

%X_transp = transpose(X_loaded);
%Y_transp = transpose(Y_loaded);

%Se revuelven los datos%

%% Set Train, test and val sets
trainRatio = 0.8;
valRatio = 0.1;
testRatio = 0.1;

[trainInd,valInd,testInd] = dividerand(size(X,2),trainRatio,valRatio,testRatio);

trainX = X(:,trainInd);
trainLabels = Y(:, trainInd);

testX = X(:,testInd);
testLabels = Y(testInd);

valX = X(:,valInd);
valLabels = Y(valInd);

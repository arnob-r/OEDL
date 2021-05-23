% This is a code of LSTM

x = load('lienard_intermittency.dat');

data=[x(:,2)]';


% Data description

numTimeStepsTrain = floor(0.9*numel(data));
dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);

% Data standarized

mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

% LSTM

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];
    
% Specify training option
    
    options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Train the LSTM network
    
net = trainNetwork(XTrain,YTrain,layers,options);
 
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

% Initialize-training and loop over

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));
numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end


 YPred = sig*YPred + mu;
 YTest = dataTest(2:end);
 rmse = sqrt(mean((YPred-YTest).^2))

%figure
%plot(dataTrain(1:end-1))
%hold on
%idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
%plot(idx,[data(numTimeStepsTrain) YPred],'.-')
%hold off
%xlabel("Time")
%ylabel("Observable")
%title("Forecast")
%legend(["Observed" "Forecast"])
%figure
%subplot(2,1,1)
%plot(YTest)
%hold on
%plot(YPred,'.-')
%hold off
%legend(["Observed" "Forecast"])
%ylabel("Cases")
%title("Forecast")
%subplot(2,1,2)
%stem(YPred - YTest)
%xlabel("Month")
%ylabel("Error")
%title("RMSE = " + rmse)

% Update Network State with observed values

net = resetState(net);
net = predictAndUpdateState(net,XTrain);

% Predict on each time step

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

% Forecasting and RMSE

YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-YTest).^2))


figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
xlabel("Time")
ylabel("Observable")
title("Forecast")

A=YTest';
B=YPred';

C=[A B];

dlmwrite('intermittency_lstm_forecast.txt',C,'delimiter','\t')    

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Time")
ylabel("Error")
title("RMSE = " + rmse)

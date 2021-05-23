% This is a code of FFNN
 
x = load('lienard_intermittency.dat');

data = [x(:,2)]';

inputData = (data(1:end-1)); 
targetData = (data(2:end));

trlen = 45000; tslen = 5000-1;
trX = inputData(1:trlen);
tsX = inputData(trlen+1:trlen+tslen);

% Remove initial points from target!
trY = targetData(1:trlen);
tsY = targetData(trlen+1:trlen+tslen);

%gradient descent with momentum
net=newff(minmax(trX),[100,1],{'logsig','purelin'},'traingdm');
%net.trainParam.show = 50;
%net.trainParam.mc = 0.9;

net.trainParam.epochs = 2500;
net.trainParam.goal = 1e-7;
net.trainParam.lr = 0.05;

[net,tr]=train(net,trX,trY);
y_net=net(trX);
%plot(trY);hold on;plot(y_net,'r--')
a=sim(net,tsX);
plot(tsY);
hold on;
plot(a,'r--')

% print the data
A1 = tsY';
B1 = a';
C1 = [A1 B1];
dlmwrite('intermittency_ffnn_forecast.txt',C1,'delimiter','\t')


RMSE = sqrt(mean((tsY - a).^2));
fprintf('Test error: %g\n', RMSE);

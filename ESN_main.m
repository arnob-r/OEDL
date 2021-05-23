% This is main code for RC

x = load('lienard_intermittency.dat');
data = x(:,2);

inputData = (data(1:end-1)); 
targetData = (data(2:end));

washout = 0;
trlen = 45000; tslen = 5000-1;
 
trX{1} = inputData(1:trlen);
tsX{1} = inputData(trlen+1:trlen+tslen);

% Remove initial points from target!
trY = targetData(1+washout:trlen);
tsY = targetData(trlen+1+washout:trlen+tslen);

% Train ESN
esn = ESN(400, 'leakRate', 0.5, 'spectralRadius', 0.001, 'regularization', 1e-8);
esn.train(trX, trY, washout);

% Test ESN
output = esn.predict(tsX, washout);

%error = immse(output, tsY);
RMSE = sqrt(mean((tsY - output).^2));

% print the data
A = tsY;
B = output;
C = [A B];

dlmwrite('intermittency_rc_forecast.txt',C,'delimiter','\t')

fprintf('Test error: %g\n', RMSE);

plot(1:length(output), output, 1:length(tsY), tsY);
legend('Output', 'Target');


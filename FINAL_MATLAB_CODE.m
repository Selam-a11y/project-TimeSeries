% USD/INR Volatility Modeling - ARIMAX-GARCH Framework
% Based on Selam Mahmud Ali Report

clear; clc;

%% 1. Load Data 
% Expected variables: Pt (prices), dates (datetime)
load('usd_inr_data.mat');

% Compute log returns
Rt = diff(log(Pt));
dates = dates(2:end);

%% 2. Structural Break Dummy (COVID-19)
covid_date = datetime(2020,3,16);
D_covid = (dates >= covid_date);

%% 3. Train/Test Split (80/20)
T = length(Rt);
split = floor(0.8 * T);

R_train = Rt(1:split);
R_test  = Rt(split+1:end);

D_train = D_covid(1:split);
D_test  = D_covid(split+1:end);

%% 4. ARIMAX(1,0,0) - GARCH(1,1)
Mdl = arima('ARLags',1,'Constant',NaN,'Beta',NaN);
Mdl.Variance = garch(1,1);

EstMdl = estimate(Mdl, R_train, 'X', D_train);

%% 5. Forecast
numSteps = length(R_test);
[forecast, forecastVar] = forecast(EstMdl, numSteps, ...
    'Y0', R_train, 'X0', D_train, 'XF', D_test);

%% 6. RMSE Calculation
rmse = sqrt(mean((R_test - forecast).^2));
disp(['Benchmark RMSE: ', num2str(rmse)]);

%% 7. EGARCH Model
Mdl_egarch = arima('ARLags',1,'Constant',NaN,'Beta',NaN);
Mdl_egarch.Variance = egarch(1,1);

EstMdl_egarch = estimate(Mdl_egarch, R_train, 'X', D_train);

[forecast_eg, var_eg] = forecast(EstMdl_egarch, numSteps, ...
    'Y0', R_train, 'X0', D_train, 'XF', D_test);

rmse_eg = sqrt(mean((R_test - forecast_eg).^2));
disp(['EGARCH RMSE: ', num2str(rmse_eg)]);

%% End of Script

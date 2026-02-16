clear;
clc;
close all;

data = readtable('uq_combined_results.csv');

%Inputs: E,rho,omega,sigma_uts
X = [data.E_Pa,data.rho_kg_m3,data.omega_rad_s,data.sigma_uts_Pa];

% Output: max von misses stress
y = data.max_mises_Pa;

X_mean =mean(X);
X_std = std(X);
X_norm = (X - X_mean) ./ X_std;

y_mean = mean(y);
y_std = std(y);
y_norm = (y - y_mean) / y_std;


norm_params.X_mean = X_mean;
norm_params.X_std = X_std;
norm_params.y_mean = y_mean;
norm_params.y_std = y_std;

rng(42);

% Number of samples
n_total = size(X_norm,1);
n_train = 80;
n_test = 20;

idx_shuffle= randperm(n_total);

% Split the data into training and testing sets
train_idx = idx_shuffle(1:n_train);
test_idx = idx_shuffle(n_train+1:end);

% Create training and testing sets
X_train = X_norm(train_idx, :);
y_train = y_norm(train_idx);
X_test = X_norm(test_idx, :);
y_test = y_norm(test_idx);
y_test_original= y(test_idx);


% Train kriging / Gaussian Process Model

fprintf('   TRAINING KRIGING MODEL\n');
tic;

gpr_model = fitrgp(X_train, y_train,'KernelFunction','ardsquaredexponential','Standardize',false);


time_kriging = toc;
fprintf('Training time: %.3f seconds\n',time_kriging);

%Predict on test set
[y_pred_krg_norm,y_std_krg_norm]= predict(gpr_model, X_test);

y_pred_krg = y_pred_krg_norm * y_std + y_mean;
y_std_krg = y_std_krg_norm * y_std;

% Calculate the root mean square error (RMSE) of the predictions
rmse_krg = sqrt(mean((y_test_original - y_pred_krg).^2));
R2_krg = 1 - sum((y_test_original - y_pred_krg).^2)/sum((y_test_original-mean(y_test_original)).^2);


fprintf('RMSE of Kriging model: %.4f MPa\n', rmse_krg/1e6);
fprintf('R^2 : %.6f\n', R2_krg);
fprintf('Mean uncertainity : %.2f MPa \n', mean(y_std_krg)/1e6);

% Train ANN 

fprintf('   TRAINING ANN MODEL\n');
tic;
hidden_layers =[10 10];
net = feedforwardnet(hidden_layers);

net.trainFcn = 'trainlm';
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-10;
net.trainParam.showWindow = true;

net.divideFcn ='dividetrain';

rng(42);
[net, tr] = train(net, X_train', y_train');

time_ann = toc;
fprintf('Training time: %.3f seconds\n', time_ann);
fprintf('Training stopped at epoch: %d\n', tr.num_epochs);

% Predict on test set using ANN
y_pred_ann_norm = net(X_test')';
y_pred_ann = y_pred_ann_norm * y_std + y_mean;

% Calculate errors
error_ann = y_test_original - y_pred_ann;
RMSE_ann = sqrt(mean(error_ann.^2));
% Calculate R^2 for ANN model
R2_ann = 1 - sum((y_test_original - y_pred_ann).^2) / sum((y_test_original - mean(y_test_original)).^2);

fprintf('\n --- ANN Results --- \n');
fprintf('RMSE: %.4f MPa\n', RMSE_ann/1e6);
fprintf('R^2 : %.6f\n',R2_ann);

 % Model Comparison
model_names = {'Kriging', 'ANN'};

% Plot RMSE and R^2 for each model
models = {'Kriging', 'ANN'};
subplot(1,3,1);
rmse_values = [rmse_krg, RMSE_ann]/1e6;
bar(rmse_values,'FaceColor',[0.3 0.5 0.8]);
set(gcf, 'Toolbar', 'none');
set(gca, 'XTickLabel',model_names,'FontSize',11);
ylabel('RMSE[MPa]','FontSize',12);
title('Model Comparison: RMSE');
grid on;

for i = 1:length(rmse_values)
    text(i,rmse_values(i) + max(rmse_values)*0.05, sprintf('%.2f',rmse_values(i)), "HorizontalAlignment","center","FontSize",10);
end

subplot(1,3,2);
r2_values = [R2_krg, R2_ann];
bar(r2_values,'FaceColor',[0.3 0.7 0.4]);

% Plot R^2 values for each model
set(gca, 'XTickLabel', model_names,'FontSize',11);
ylabel('R^2 Value','FontSize',12);
title('Model Comparison: R^2 Values');
ylim([min(r2_values)*0.99,1.001]);
grid on;

for i = 1:length(r2_values)
    text(i,r2_values(i) + 0.001, sprintf('%.2f',r2_values(i)), "HorizontalAlignment","center","FontSize",10);
end

subplot(1,3,3);
hold on;

actual_range = [min(y_test_original), max(y_test_original)]/1e6;
plot_range = [actual_range(1)-10, actual_range(2)+10];
plot(plot_range,plot_range,'k--','LineWidth',2,'DisplayName','Prediction');

scatter(y_test_original/1e6, y_pred_krg/1e6, 100,'g', 'filled', 'DisplayName', 'Kriging');
scatter(y_test_original/1e6, y_pred_ann/1e6, 100,'r' , 'filled', 'DisplayName', 'ANN');


xlabel('Actual Stress [MPa]', 'FontSize', 12);
ylabel('Predicted Stress [MPa]', 'FontSize', 12);
title('Predicted vs Actual', 'FontSize', 13);
legend('Location', 'northwest', 'FontSize', 10);
grid on;
axis equal;
xlim(plot_range);
ylim(plot_range);

% Save figure
sgtitle('Surrogate Model Comparison for Rotating Disk', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'surrogate_comparison.png');


fprintf('   RELIABILITY ANALYSIS (KRIGING)\n');
% Number of Monte Carlo samples
n_mc = 100000;

% Material parameters (Aluminum 2618-T61)
E_mean = 71e9;       E_std = 2e9;
rho_mean = 2900;     rho_std = 30;
sigma_uts_mean = 420e6;  sigma_uts_std = 40e6;

% Current design: 20,000 RPM
omega_rpm_current = 20000;
omega_current = omega_rpm_current * (2*pi/60);
omega_std = 700 * (2*pi/60);

% Generate random samples
rng(123);
E_mc = normrnd(E_mean, E_std, n_mc, 1);
rho_mc = normrnd(rho_mean, rho_std, n_mc, 1);
omega_mc = normrnd(omega_current, omega_std, n_mc, 1);
sigma_uts_mc = normrnd(sigma_uts_mean, sigma_uts_std, n_mc, 1);

% Combine into input matrix
X_mc = [E_mc, rho_mc, omega_mc, sigma_uts_mc];

% Normalize using same parameters as training
X_mc_norm = (X_mc - norm_params.X_mean) ./ norm_params.X_std;

% Predict stress using Kriging
tic;
[y_mc_norm, ~] = predict(gpr_model, X_mc_norm);
time_predict = toc;
fprintf('Prediction time: %.2f seconds\n', time_predict);

% Denormalize predictions
stress_mc = y_mc_norm * norm_params.y_std + norm_params.y_mean;

% Calculate limit state: g = strength - stress
g_mc = sigma_uts_mc - stress_mc;

% Count failures (g < 0 means failure)
n_fail = sum(g_mc < 0);
P_f = n_fail / n_mc;

% Calculate reliability index
beta = -norminv(P_f);

fprintf('\n=== RELIABILITY RESULTS ===\n');
fprintf('Samples: %d\n', n_mc);
fprintf('Failures: %d\n', n_fail);
fprintf('P_f = %.4f (%.2f%%)\n', P_f, P_f*100);
fprintf('Reliability Index β = %.4f\n', beta);


fprintf('   OPTIMIZATION: RPM vs P_f CURVE\n');

RPM_range = 10000:500:22000;
n_rpm = length(RPM_range);

% Store results
P_f_curve = zeros(n_rpm, 1);


for i = 1:n_rpm
    % Current RPM
    rpm_test = RPM_range(i);
    omega_test = rpm_test * (2*pi/60);
    
    % Generate MC samples for this RPM
    omega_mc_test = normrnd(omega_test, omega_std, n_mc, 1);
    
    % Create input matrix
    X_mc_test = [E_mc, rho_mc, omega_mc_test, sigma_uts_mc];
    
    % Normalize
    X_mc_test_norm = (X_mc_test - norm_params.X_mean) ./ norm_params.X_std;
    
    % Predict stress
    [y_mc_test_norm, ~] = predict(gpr_model, X_mc_test_norm);
    stress_mc_test = y_mc_test_norm * norm_params.y_std + norm_params.y_mean;
    
    % Calculate P_f
    g_test = sigma_uts_mc - stress_mc_test;
    P_f_curve(i) = sum(g_test < 0) / n_mc;
    
    % Progress
    if mod(i, 4) == 0
        fprintf('  %d/%d complete (RPM=%d, P_f=%.2f%%)\n', i, n_rpm, rpm_test, P_f_curve(i)*100);
    end
end

% Find critical RPM values using proper interpolation
fprintf('\n=== CRITICAL RPM VALUES ===\n');

% Convert to percentages for easier handling
P_f_percent = P_f_curve * 100;

% Find RPM for P_f = 10%
idx_below_10 = find(P_f_percent < 10, 1, 'last');
idx_above_10 = find(P_f_percent >= 10, 1, 'first');
if ~isempty(idx_below_10) && ~isempty(idx_above_10) && idx_above_10 > idx_below_10
    RPM_10 = interp1(P_f_percent(idx_below_10:idx_above_10), ...
                     RPM_range(idx_below_10:idx_above_10), 10, 'linear');
    fprintf('Safe RPM (P_f = 10%%): %.0f RPM\n', RPM_10);
else
    RPM_10 = NaN;
    fprintf('Safe RPM (P_f = 10%%): Cannot determine (outside range)\n');
end

% Find RPM for P_f = 50%
idx_below_50 = find(P_f_percent < 50, 1, 'last');
idx_above_50 = find(P_f_percent >= 50, 1, 'first');
if ~isempty(idx_below_50) && ~isempty(idx_above_50) && idx_above_50 > idx_below_50
    RPM_50 = interp1(P_f_percent(idx_below_50:idx_above_50), ...
                     RPM_range(idx_below_50:idx_above_50), 50, 'linear');
    fprintf('Critical RPM (P_f = 50%%): %.0f RPM\n', RPM_50);
else
    RPM_50 = NaN;
    fprintf('Critical RPM (P_f = 50%%): Cannot determine (outside range)\n');
end

% Find RPM for P_f = 90%
idx_below_90 = find(P_f_percent < 90, 1, 'last');
idx_above_90 = find(P_f_percent >= 90, 1, 'first');
if ~isempty(idx_below_90) && ~isempty(idx_above_90) && idx_above_90 > idx_below_90
    RPM_90 = interp1(P_f_percent(idx_below_90:idx_above_90), ...
                     RPM_range(idx_below_90:idx_above_90), 90, 'linear');
    fprintf('Burst RPM (P_f = 90%%): %.0f RPM\n', RPM_90);
else
    RPM_90 = NaN;
    fprintf('Burst RPM (P_f = 90%%): Cannot determine (outside range)\n');
end

fprintf('Current Design (20,000 RPM): P_f = %.2f%%\n', P_f*100);

% Plot
figure('Position', [100, 100, 800, 500], 'Color', 'w');

plot(RPM_range, P_f_curve*100, 'b-', 'LineWidth', 2);
hold on;

% Mark critical points (if they exist)
if ~isnan(RPM_10)
    plot(RPM_10, 10, 'go', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'LineWidth', 2);
end
if ~isnan(RPM_50)
    plot(RPM_50, 50, 'yo', 'MarkerSize', 12, 'MarkerFaceColor', 'y', 'LineWidth', 2);
end
if ~isnan(RPM_90)
    plot(RPM_90, 90, 'ro', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'LineWidth', 2);
end
plot(20000, P_f*100, 'ms', 'MarkerSize', 14, 'MarkerFaceColor', 'm', 'LineWidth', 2);

% Reference lines
yline(10, 'g--', 'LineWidth', 1);
yline(50, 'y--', 'LineWidth', 1);
yline(90, 'r--', 'LineWidth', 1);
xline(20000, 'm--', 'LineWidth', 1);

xlabel('Rotational Speed [RPM]', 'FontSize', 12);
ylabel('Probability of Failure P_f [%]', 'FontSize', 12);
title('Aluminum Flywheel: RPM vs Failure Probability', 'FontSize', 14, 'FontWeight', 'bold');

% Dynamic legend
legend_entries = {'P_f Curve'};
if ~isnan(RPM_10), legend_entries{end+1} = sprintf('Safe (%.0f RPM)', RPM_10); end
if ~isnan(RPM_50), legend_entries{end+1} = sprintf('Critical (%.0f RPM)', RPM_50); end
if ~isnan(RPM_90), legend_entries{end+1} = sprintf('Burst (%.0f RPM)', RPM_90); end
legend_entries{end+1} = sprintf('Current Design (%.1f%%)', P_f*100);
legend(legend_entries, 'Location', 'northwest');

grid on;
xlim([min(RPM_range), max(RPM_range)]);
ylim([0, 100]);

saveas(gcf, 'rpm_vs_pf_curve.png');
fprintf('\nFigure saved: rpm_vs_pf_curve.png\n');

fprintf('   FINAL SUMMARY\n');
fprintf('\n--- Design Points ---\n');
fprintf('%-25s %10s %10s\n', 'Description', 'RPM', 'P_f');
fprintf('%-25s %10s %10s\n', '-----------', '---', '---');
fprintf('%-25s %10.0f %9.1f%%\n', 'Safe (P_f = 10%)', RPM_10, 10);
fprintf('%-25s %10.0f %9.1f%%\n', 'Critical (P_f = 50%)', RPM_50, 50);
fprintf('%-25s %10.0f %9.1f%%\n', 'Burst (P_f = 90%)', RPM_90, 90);
fprintf('%-25s %10d %9.1f%%\n', 'Current Design', 20000, P_f*100);

fprintf('\n--- Surrogate Performance ---\n');
fprintf('Model: Kriging (Gaussian Process)\n');
fprintf('RMSE: %.4f MPa\n', rmse_krg/1e6);
fprintf('R²: %.6f\n', R2_krg);
fprintf('MC Samples: %d\n', n_mc);
fprintf('Prediction Time: %.2f seconds\n', time_predict);
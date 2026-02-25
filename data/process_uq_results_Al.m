%% PROCESS UQ RESULTS - Statistics, Sensitivity, and Visualization
close all
clearvars
clc

%% LOAD DATA
fprintf('Loading results...\n');

data = readtable('uq_combined_results.csv');
E_samples = data.E_Pa;
rho_samples = data.rho_kg_m3;
omega_samples = data.omega_rad_s;
sigma_uts_samples = data.sigma_uts_Pa;
max_mises_results = data.max_mises_Pa;

N_samples = length(max_mises_results);
valid_idx = ~isnan(max_mises_results);
N_valid = sum(valid_idx);
mises_valid = max_mises_results(valid_idx);

fprintf('Loaded %d samples (%d valid)\n\n', N_samples, N_valid);

%% STRESS STATISTICS
mean_mises = mean(mises_valid);
std_mises = std(mises_valid);
p95 = prctile(mises_valid, 95);

fprintf('=== STRESS STATISTICS ===\n');
fprintf('Mean:  %.2f MPa\n', mean_mises/1e6);
fprintf('Std:   %.2f MPa\n', std_mises/1e6);
fprintf('CoV:   %.2f%%\n', std_mises/mean_mises*100);
fprintf('95th:  %.2f MPa\n\n', p95/1e6);

%% SAFETY FACTOR
safety_factors = sigma_uts_samples(valid_idx) ./ mises_valid;
mean_sf = mean(safety_factors);
prob_failure = sum(safety_factors < 1.0) / N_valid * 100;

fprintf('=== SAFETY FACTOR ===\n');
fprintf('Mean SF: %.2f\n', mean_sf);
fprintf('Min SF:  %.2f\n', min(safety_factors));
fprintf('P(SF<1): %.2f%%\n\n', prob_failure);

%% SENSITIVITY ANALYSIS
X_valid = [E_samples(valid_idx), rho_samples(valid_idx), ...
           omega_samples(valid_idx), sigma_uts_samples(valid_idx)];
rho_corr = corr(X_valid, mises_valid, 'Type', 'Spearman');
param_names = {'E', 'rho', 'omega', 'sigma_uts'};

fprintf('=== SENSITIVITY (Spearman) ===\n');
for i = 1:4
    fprintf('%s: %+.3f\n', param_names{i}, rho_corr(i));
end
fprintf('\n');

%% VISUALIZATION
figure('Position', [100, 100, 1200, 400]);

% Stress distribution
subplot(1,3,1);
histogram(mises_valid/1e6, 20, 'FaceColor', [0.2 0.5 0.8], 'Normalization', 'pdf');
hold on; xline(mean_mises/1e6, 'r-', 'LineWidth', 2);
xlabel('Max von Mises [MPa]'); ylabel('PDF');
title(sprintf('Stress (\\mu=%.1f, \\sigma=%.1f MPa)', mean_mises/1e6, std_mises/1e6));
grid on;

% Safety factor
subplot(1,3,2);
histogram(safety_factors, 20, 'FaceColor', [0.8 0.4 0.2], 'Normalization', 'pdf');
hold on; xline(1.0, 'k--', 'LineWidth', 2);
xlabel('Safety Factor'); ylabel('PDF');
title(sprintf('SF (\\mu=%.2f, P(fail)=%.1f%%)', mean_sf, prob_failure));
grid on;

% Sensitivity
subplot(1,3,3);
bar(rho_corr);
set(gca, 'XTickLabel', param_names);
ylabel('Spearman Correlation'); ylim([-1 1]);
title('Sensitivity Analysis: Aluminium');
grid on;

saveas(gcf, 'uq_results.png');
fprintf('Saved: uq_results.png\n');

%% SCATTER PLOTS
figure('Position', [100, 100, 800, 600]);

subplot(2,2,1);
scatter(omega_samples(valid_idx), mises_valid/1e6, 20, 'filled');
xlabel('\omega [rad/s]'); ylabel('Stress [MPa]');
title(sprintf('\\rho = %.3f', rho_corr(3))); grid on;

subplot(2,2,2);
scatter(rho_samples(valid_idx), mises_valid/1e6, 20, 'filled');
xlabel('\rho [kg/m³]'); ylabel('Stress [MPa]');
title(sprintf('\\rho = %.3f', rho_corr(2))); grid on;

subplot(2,2,3);
scatter(E_samples(valid_idx)/1e9, mises_valid/1e6, 20, 'filled');
xlabel('E [GPa]'); ylabel('Stress [MPa]');
title(sprintf('\\rho = %.3f', rho_corr(1))); grid on;

subplot(2,2,4);
scatter(sigma_uts_samples(valid_idx)/1e6, safety_factors, 20, 'filled');
xlabel('\sigma_{uts} [MPa]'); ylabel('Safety Factor');
hold on; yline(1, 'r--'); grid on;

saveas(gcf, 'uq_scatter.png');
fprintf('Saved: uq_scatter.png\n');

%% SAVE RESULTS
uq_results.stats = struct('mean_mises', mean_mises, 'std_mises', std_mises, ...
    'mean_sf', mean_sf, 'prob_failure', prob_failure);
uq_results.sensitivity = rho_corr;
save('uq_processed_results.mat', 'uq_results');
fprintf('Saved: uq_processed_results.mat\n');

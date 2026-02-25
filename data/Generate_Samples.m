%% STEP 1: GENERATE LHS SAMPLES AND CREATE .INP FILES
% Rotating Disk UQ Analysis - Aluminum Flywheel
% This script generates Latin Hypercube Samples and creates Abaqus input files

clear; clc; close all;

%% CONFIGURATION
template_inp = 'template_Al.inp'; 
N_samples = 100;                      


% Line numbers in template 
LINE_DENSITY = 2153;
LINE_ELASTIC = 2155;
LINE_CENTRIF = 2174;

% Fixed parameter
nu = 0.33;  % Poisson's ratio

%% MATERIAL PARAMETERS (Aluminum 2618-T61)
% Uncertain parameters with Normal distributions

% Young's Modulus
E_mean = 71e9;      % Pa
E_std = 2e9;        % Pa 

% Density
rho_mean = 2900;    % kg/m³
rho_std = 30;       % kg/m³ 

% Ultimate Tensile Strength
sigma_uts_mean = 420e6;   % Pa
sigma_uts_std = 40e6;     % Pa 

% Angular Velocity (20,000 RPM nominal)
omega_rpm_mean = 20000;   % RPM
omega_rpm_std = 700;      % RPM 

% Convert to rad/s
omega_mean = omega_rpm_mean * (2*pi/60);  % rad/s
omega_std = omega_rpm_std * (2*pi/60);    % rad/s


%% GENERATE LHS SAMPLES
rng(42, 'twister');  

% Generate LHS in [0,1] space
lhs_uniform = lhsdesign(N_samples, 4, 'criterion', 'maximin');

% Transform to Normal distributions using inverse CDF

E_samples = E_mean + E_std * norminv(lhs_uniform(:,1));
rho_samples = rho_mean + rho_std * norminv(lhs_uniform(:,2));
omega_samples = omega_mean + omega_std * norminv(lhs_uniform(:,3));
sigma_uts_samples = sigma_uts_mean + sigma_uts_std * norminv(lhs_uniform(:,4));

% Combine into matrix
X_samples = [E_samples, rho_samples, omega_samples, sigma_uts_samples];

% Display ranges
fprintf('\nGenerated Sample Ranges:\n');
fprintf('  E:     [%.2f, %.2f] GPa\n', min(E_samples)/1e9, max(E_samples)/1e9);
fprintf('  rho:   [%.1f, %.1f] kg/m³\n', min(rho_samples), max(rho_samples));
fprintf('  omega: [%.1f, %.1f] rad/s = [%.0f, %.0f] RPM\n', ...
    min(omega_samples), max(omega_samples), ...
    min(omega_samples)*60/(2*pi), max(omega_samples)*60/(2*pi));
fprintf('  sigma_uts: [%.1f, %.1f] MPa\n', min(sigma_uts_samples)/1e6, max(sigma_uts_samples)/1e6);

%% READ TEMPLATE FILE

fid = fopen(template_inp, 'r');
if fid == -1
    error('Cannot open template file: %s', template_inp);
end

template_lines = {};
while ~feof(fid)
    template_lines{end+1} = fgetl(fid);
end
fclose(fid);

fprintf('  Template has %d lines\n', length(template_lines));

% Verify line numbers
fprintf('\nVerifying line numbers:\n');
fprintf('  Line %d: %s\n', LINE_DENSITY, template_lines{LINE_DENSITY});
fprintf('  Line %d: %s\n', LINE_ELASTIC, template_lines{LINE_ELASTIC});
fprintf('  Line %d: %s\n', LINE_CENTRIF, template_lines{LINE_CENTRIF});

%% CREATE OUTPUT DIRECTORY
output_dir = 'inp_files';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% CREATE MODIFIED .INP FILES
fprintf('\nCreating %d .inp files...\n', N_samples);

for k = 1:N_samples
    E_k = X_samples(k, 1);
    rho_k = X_samples(k, 2);
    omega_k = X_samples(k, 3);
    sigma_uts_k = X_samples(k, 4);
    
    
    omega_sq_k = omega_k^2;
    modified_lines = template_lines;
    modified_lines{LINE_DENSITY} = sprintf('%.4f,', rho_k);
    modified_lines{LINE_ELASTIC} = sprintf(' %.6e, %.2f', E_k, nu);
    modified_lines{LINE_CENTRIF} = sprintf('Set-2, CENTRIF, %.6e,0.,0.,0.,0.0,1.0,0.0', omega_sq_k);
    
    % Write .inp file
    inp_filename = fullfile(output_dir, sprintf('RotatingDisk_Sample_%d.inp', k));
    fid = fopen(inp_filename, 'w');
    for j = 1:length(modified_lines)
        if isnumeric(modified_lines{j}) && modified_lines{j} == -1
            break;
        end
        fprintf(fid, '%s\n', modified_lines{j});
    end
    fclose(fid);
    
    % Progress
    if mod(k, 10) == 0
        fprintf('  Created %d/%d files\n', k, N_samples);
    end

end
fprintf('  All .inp files created in: %s/\n', output_dir);

%% CREATE BATCH FILE FOR ABAQUS

batch_filename = fullfile(output_dir, 'Run_All_Simulations.bat');
fid = fopen(batch_filename, 'w');

fprintf(fid, 'echo    Running %d Abaqus Simulations\n', N_samples);

for k = 1:N_samples
    fprintf(fid, 'echo Running Sample %d of %d...\n', k, N_samples);
    fprintf(fid, 'call abaqus job=RotatingDisk_Sample_%d interactive\n', k);
    fprintf(fid, 'echo.\n');
end

fprintf(fid, 'echo    All %d simulations completed!\n', N_samples);
fclose(fid);
%% CREATE INPUT FILES FOR PYTHON EXTRACTION

for k = 1:N_samples
    inp_filename = fullfile(output_dir, sprintf('Inputs_Sample_%d.txt', k));
    fid = fopen(inp_filename, 'w');
    fprintf(fid, '%.6e\n', X_samples(k,1));  % E
    fprintf(fid, '%.4f\n', X_samples(k,2));  % rho
    fprintf(fid, '%.6f\n', X_samples(k,3));  % omega
    fprintf(fid, '%.6e\n', X_samples(k,4));  % sigma_uts
    fprintf(fid, '%.6e\n', X_samples(k,3)^2); % omega_sq
    fclose(fid);
end


%% SAVE SAMPLE DATA
sample_data.N_samples = N_samples;
sample_data.X_samples = X_samples;
sample_data.E_samples = E_samples;
sample_data.rho_samples = rho_samples;
sample_data.omega_samples = omega_samples;
sample_data.sigma_uts_samples = sigma_uts_samples;
sample_data.param_names = {'E_Pa', 'rho_kg_m3', 'omega_rad_s', 'sigma_uts_Pa'};
sample_data.line_numbers = struct('density', LINE_DENSITY, 'elastic', LINE_ELASTIC, 'centrif', LINE_CENTRIF);

save('sample_data.mat', 'sample_data');



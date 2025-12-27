%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%
%                 Project Work: Financial Econometrics                    %
%
%  University of Bologna — LM(EC)² 
%  Last Update:  3/12/2025
%
%    • Filippo Nardoni  ~ filippo.nardoni@studio.unibo.it
%
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%


% --------------------------------------------------------------------- %%
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
%                            DAILY ANALYSIS                              %
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
%  --------------------------------------------------------------------  %

%% -------------------------------------------------------------------- %%
%                  SECTION 1: WEEK (1 and 2) ANALYSIS                    %
%  --------------------------------------------------------------------  %
clear all, clc;

file_path = "/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Financial Econometrics/Project_Nardoni";
tables_path = fullfile(file_path, "Tables/Daily");
figures_path = fullfile(file_path, "Figures/Daily");



% Data Download
data_raw = readtable(fullfile(file_path, "/Data/DEXUSEU.csv"));
data = table2array(data_raw(:,2));
time = datetime(data_raw{:,1}, 'InputFormat', 'yyyy-MM-dd');
T = size(data,1);

% Count the number of NaN values in the data
nan_count = sum(isnan(data));
fprintf('Number of NaN values in the data: %d\n', nan_count);

% Using function fillmissing using mov mean averaged at 5
data = fillmissing(data, 'movmean', 5);

% Define crisis periods
financial_crisis_start = datetime(2007, 12, 1);
financial_crisis_end = datetime(2009, 6, 30);
covid_crisis_start = datetime(2020, 3, 1);
covid_crisis_end = datetime(2020, 12, 31);

% ______ First plot
figure('Position', [100, 100, 900, 500]);

% Add shaded regions for crises FIRST (so they appear in background)
hold on;
% Financial Crisis (2007-2009)
fill([financial_crisis_start, financial_crisis_end, financial_crisis_end, financial_crisis_start], ...
     [0.5, 0.5,2,2], ...
     [0.4 0.65 0.90], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
     'DisplayName', 'Financial Crisis');

% COVID-19 Crisis (2020)
fill([covid_crisis_start, covid_crisis_end, covid_crisis_end, covid_crisis_start], ...
     [0.5, 0.5,2,2], ...
     [0.2 0.35 0.90], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
     'DisplayName', 'COVID-19 Crisis');

% Plot exchange rate on top
plot(time, data, 'b-', 'LineWidth', 1.5, 'DisplayName', 'EUR/USD');

grid on;
xlabel('Time', 'FontSize', 11);
ylabel('Exchange Rate', 'FontSize', 11);
title('EUR/USD Exchange Rate with Crisis Periods', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');
hold off;

saveas(gcf, fullfile(figures_path, 'exchange_rate.png'));


% second plot
data_diff = diff(log(data))*100;

figure('Position', [100, 100, 900, 500]);

% Add shaded regions for crises FIRST
hold on;
% Financial Crisis (2007-2009)
y_lim = [min(data_diff)*1.2, max(data_diff)*1.2];
fill([financial_crisis_start, financial_crisis_end, financial_crisis_end, financial_crisis_start], ...
     [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
     [0.6, 0.8, 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
     'DisplayName', 'Financial Crisis');

% COVID-19 Crisis (2020)
fill([covid_crisis_start, covid_crisis_end, covid_crisis_end, covid_crisis_start], ...
     [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
     [0.2 0.35 0.90], 'FaceAlpha', 0.3, 'EdgeColor', 'none', ...
     'DisplayName', 'COVID-19 Crisis');

% Plot log returns
plot(time(2:end), data_diff, 'b-', 'LineWidth', 1, 'DisplayName', 'Log Returns');

% Add mean line
mean_log_returns = mean(data_diff);
yline(mean_log_returns, 'r--', 'LineWidth', 1.5, ...
      'Label', sprintf('Mean: %.3f%%', mean_log_returns), ...
      'LabelHorizontalAlignment', 'left', ...
      'DisplayName', 'Mean');

grid on;
xlabel('Time', 'FontSize', 11);
ylabel('Log Returns (%)', 'FontSize', 11);
title('EUR/USD Log Returns with Crisis Periods', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');
ylim(y_lim);
hold off;

saveas(gcf, fullfile(figures_path, 'log_returns.png'));



%________________________ RANDOM WALK ANALYSIS _________________________ %

% Initialize results storage
rand_walk_test_exc = NaN(5,3);
rand_walk_test_diff = NaN(5,3);


% 1] Dickey-Fuller test 
[h_df, pValue_df, stat_df, criticalValues_df] = adftest(data, "Lags", 0);
rand_walk_test_exc(1,:) = [pValue_df, stat_df, criticalValues_df];

% 2] DF with deterministic trend
[h_trend, pValue_trend, stat_trend, cValue_trend] = adftest(data, "Model", "TS", "Lags", 0);
rand_walk_test_exc(2,:) = [pValue_trend, stat_trend, cValue_trend];

% 3] Augmented Dickey-Fuller test with 5 lags
[h_adf5, pValue_adf5, stat_adf5, criticalValues_adf5] = adftest(data, "Lags", 5);
rand_walk_test_exc(3,:) = [pValue_adf5, stat_adf5, criticalValues_adf5];

% 4] Kwiatkowski-Phillips-Schmidt-Shin test
[h_kpss, pValue_kpss, stat_kpss, criticalValues_kpss] = kpsstest(data);
rand_walk_test_exc(4,:) = [pValue_kpss, stat_kpss, criticalValues_kpss];

% 5] Variance Ratio Test
[h_vr, pValue_vr, stat_vr, cValue_vr] = vratiotest(data);
rand_walk_test_exc(5,:) = [pValue_vr, stat_vr, cValue_vr];


% 1] Dickey-Fuller test 
[h_df_diff, pValue_df_diff, stat_df_diff, criticalValues_df_diff] = adftest(data_diff, "Lags", 0);
rand_walk_test_diff(1,:) = [pValue_df_diff, stat_df_diff, criticalValues_df_diff];

% 2] DF with deterministic trend
[h_trend_diff, pValue_trend_diff, stat_trend_diff, cValue_trend_diff] = adftest(data_diff, "Model", "TS", "Lags", 0);
rand_walk_test_diff(2,:) = [pValue_trend_diff, stat_trend_diff, cValue_trend_diff];

% 3] Augmented Dickey-Fuller test with 5 lags
[h_adf5_diff, pValue_adf5_diff, stat_adf5_diff, criticalValues_adf5_diff] = adftest(data_diff, "Lags", 5);
rand_walk_test_diff(3,:) = [pValue_adf5_diff, stat_adf5_diff, criticalValues_adf5_diff];

% 4] Kwiatkowski-Phillips-Schmidt-Shin test
[h_kpss_diff, pValue_kpss_diff, stat_kpss_diff, criticalValues_kpss_diff] = kpsstest(data_diff);
rand_walk_test_diff(4,:) = [pValue_kpss_diff, stat_kpss_diff, criticalValues_kpss_diff];

% 5] Variance Ratio Test
[h_vr_diff, pValue_vr_diff, stat_vr_diff, cValue_vr_diff] = vratiotest(data_diff);
rand_walk_test_diff(5,:) = [pValue_vr_diff, stat_vr_diff, cValue_vr_diff];


test_n_exc = ["DF", "DF_Trend", "ADF_5", "KPSS", "VAR_Ratio"];
test_n_diff = ["DF", "DF_Trend", "ADF_5", "KPSS", "VAR_Ratio"];
metric_n = ["p_value", "TestStatistic", "CriticalValue"];


table_rand_walk_exc = array2table(rand_walk_test_exc', "RowNames", metric_n,"VariableNames", test_n_exc);

table_rand_walk_diff = array2table(rand_walk_test_diff', "RowNames", metric_n, "VariableNames", test_n_diff);

% Display tables
disp('Unit Root Tests - Levels (Tests as Columns):');
disp(table_rand_walk_exc);
disp(' ');
disp('Unit Root Tests - First Differences (Tests as Columns):');
disp(table_rand_walk_diff);

% Save tables
writetable(table_rand_walk_exc, fullfile(tables_path, 'unit_root_tests_levels.txt'), 'WriteRowNames', true);
writetable(table_rand_walk_diff, fullfile(tables_path, 'unit_root_tests_diff.txt'), 'WriteRowNames', true);

% Optional: Save to CSV for easier reading
writetable(table_rand_walk_exc, fullfile(tables_path, 'unit_root_tests_levels.csv'), 'WriteRowNames', true);
writetable(table_rand_walk_diff, fullfile(tables_path, 'unit_root_tests_diff.csv'), 'WriteRowNames', true);


%________________________ STATIONARITY ANALYSIS _________________________ %


% Correlogram function called
c = correlogram(data, 100, 1);
saveas(gcf, fullfile(figures_path, 'correlogram_levels.png'));

d = correlogram(data_diff, 100, 1);
saveas(gcf, fullfile(figures_path, 'correlogram_diff.png'));

d = correlogram(data_diff, 100, 1, "^2", "^2");
saveas(gcf, fullfile(figures_path, 'correlogram_diff_sqrt.png'));


% Short Long run dependence Poterba-Summers
beta = PS_function(data, 1, 100);
figure;
plot(beta, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Lag');
ylabel('Beta');
title('Poterba-Summers Test');
saveas(gcf, fullfile(figures_path, 'poterba_summers.png'));


%%_________________________ MOMENTS ANALYSIS ___________________________ %

% Report the Mandelbrot analysis
var_analysis = mand_function(data_diff, 1);

moments_table = table({'Mean'; 'Std Dev'; 'Skewness'; 'Kurtosis'}, ...
                      [mu; sigma; sym; kurt], ...
                      'VariableNames', {'Statistic', 'Value'})
writetable(moments_table, fullfile(tables_path, 'descriptive_stats.csv'));

%%________________________ NORMALITY ANALYSIS __________________________ %

z_data = zscore(data_diff);

figure;
hold on;
histogram(z_data, 100, "Normalization", "pdf", "FaceAlpha", 0.4, "FaceColor", 'b');

x = linspace(-4, 4, 1000);
y = normpdf(x, 0, 1);


plot(x, y, 'r-', 'LineWidth', 1.8);
xline(3, 'k--', 'LineWidth', 1.5);
xline(-3, 'k--', 'LineWidth', 1.5);

[f, xi, bw] = ksdensity(z_data)
plot(xi, f, 'b-', 'LineWidth', 2);

extreme_idx = abs(z_data) > 3;
extreme_count_right = sum(z_data > 3);
extreme_count_left = sum(z_data < -3);

plot(z_data(extreme_idx), zeros(sum(extreme_idx),1), 'ro', 'MarkerSize', 6, 'LineWidth', 1.2);

xlabel("Standardized Returns");
ylabel("Density");
title("Histogram of Standardized Returns with N(0,1) Overlay");
legend({"Empirical PDF", "N(0,1)", "z=±3", "", "Kernel density", "Extreme values"}, 'Location', 'best');
grid on;
hold off;
saveas(gcf, fullfile(figures_path, 'normality_histogram.png'));

% Test on Normality
[h_jb, p_jb, jbStat] = jbtest(z_data);
[h_ks, p_ks, ksStat] = kstest(z_data); 
[h_sw, p_sw] = swtest(z_data);
[h_cvm, p_cvm, A2_stat, ~] = cvmtest_known_normal(z_data);

normality_tests_table = table( ...
    {'Jarque-Bera'; 'Kolmogorov-Smirnov'; 'Shapiro-Wilk'; 'Cramér–von Mises'}, ...
    [jbStat; ksStat; NaN; A2_stat], ...
    [p_jb; p_ks; p_sw; p_cvm], ...
    [h_jb; h_ks; h_sw; h_cvm], ...
    'VariableNames', {'Test', 'Statistic', 'p_value', 'Reject_H0'} ...
);
% Save the normality tests results to a CSV file
writetable(normality_tests_table, fullfile(tables_path, 'normality_tests_summary.csv'));



alpha = Hill_est(data_diff, extreme_count_left, "SX");











%% -------------------------------------------------------------------- %%
%                   SECTION 2: WEEK (2) ANALYSIS                         %
%  --------------------------------------------------------------------  %

% ________________ ESTIMATION GARCH(P,Q) MODELS _________________________ %

wind = floor(length(data_diff)/3)*2;
P = 10;
Q = 10;

aic_pq = zeros(P, Q);
bic_pq = zeros(P, Q);

min_aic = 99999999;
index_aic_min = NaN(2, 1);

min_bic = 99999999;
index_bic_min = NaN(2, 1);

for p = 1:P
    for q = 1:Q
        try
            Mdl = garch(p, q);
            [EstMdl, EstParamCov, logL] = estimate(Mdl, data_diff(1:wind), 'Display', 'off');

            k = 1 + p + q;
            T = length(data_diff);

            AIC = -2*logL + 2*k;
            BIC = -2*logL + k*log(T);

            aic_pq(p, q) = AIC;
            bic_pq(p, q) = BIC;

            if aic_pq(p,q) < min_aic
                min_aic = aic_pq(p,q);
                index_aic_min = [p, q];
            end

            if bic_pq(p,q) < min_bic
                min_bic = bic_pq(p,q);
                index_bic_min = [p, q];
            end

        catch
            aic_pq(p, q) = NaN;
            bic_pq(p, q) = NaN;
        end
    end
end

% Save AIC and BIC matrices
writematrix(aic_pq, fullfile(tables_path, 'aic_matrix.csv'));
writematrix(bic_pq, fullfile(tables_path, 'bic_matrix.csv'));

figure;
surf(aic_pq);
colormap("Sky");
hold on;
xlabel('q (ARCH lags)');
ylabel('p (GARCH lags)');
zlabel('AIC');
title('AIC Surface for GARCH(p,q)');

pA = index_aic_min(1);
qA = index_aic_min(2);
zA = aic_pq(pA, qA);
plot3(qA, pA, zA, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
saveas(gcf, fullfile(figures_path, 'aic_surface.png'));

figure;
surf(bic_pq);
colormap("sky");
hold on;
xlabel('q (ARCH lags)');
ylabel('p (GARCH lags)');
zlabel('BIC');
title('BIC Surface for GARCH(p,q)');

pB = index_bic_min(1);
qB = index_bic_min(2);
zB = bic_pq(pB, qB);
plot3(qB, pB, zB, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
saveas(gcf, fullfile(figures_path, 'bic_surface.png'));



% Model selected
p = pB;
q = qB;
Mdl = garch(p, q);
[EstMdl, EstParamCov, logL] = estimate(Mdl, data_diff(1:wind), 'Display', 'off');

alpha = EstMdl.ARCH{1};
beta = EstMdl.GARCH{1};
omega = EstMdl.Constant;


se = sqrt(diag(EstParamCov));

% Calculate t-statistics
coef_values = [omega; alpha; beta];
t_stats = coef_values ./ se;


p_values = 2 * (1 - normcdf(abs(t_stats)));


garch_params = table({'omega'; 'alpha'; 'beta'}, ...
                      coef_values, ...
                      se, ...
                      t_stats, ...
                      p_values, ...
                      'VariableNames', {'Parameter', 'Coefficient', 'Std_Error', 't_Statistic', 'p_Value'});

disp('GARCH Model Parameter Estimates:');
disp(garch_params);



% Continue with variance calculation
x2 = data_diff(1:wind).^2;
sigma2 = zeros(length(x2)+1, 1);
sigma2(1) = omega / (1 - alpha - beta);
T = length(x2);
for i = 1:T
    sigma2(i+1) = omega + alpha * x2(i) + beta * sigma2(i);
end

figure;
plot(time(1:wind), sigma2(2:end), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time');
ylabel('Conditional Variance');
title('GARCH(p,q) Conditional Variance');
saveas(gcf, fullfile(figures_path, 'garch_variance.png'));

% Normality Analysis of z_t
z_t = data_diff(1:wind) ./ sqrt(sigma2(2:end));
[h_jb, pValue_jb, jbStat] = jbtest(z_t);
[h_cvm, pValue_cvm, A2_stat, ~] = cvmtest_known_normal(z_t);
[h_sw, pValue_sw] = swtest(z_t);

TestNames = {'Jarque-Bera'; 'Cramér-von Mises'; 'Shapiro-Wilk'};
TestStatistic = [jbStat; A2_stat; NaN];
PValue = [pValue_jb; pValue_cvm; pValue_sw];
RejectH0 = logical([h_jb; h_cvm; h_sw]);
Conclusion = cell(3,1);
for i = 1:3
    if RejectH0(i)
        Conclusion{i} = 'Non-Normal';
    else
        Conclusion{i} = 'Normal';
    end
end

NormalityTests = table(TestNames, TestStatistic, PValue, RejectH0, Conclusion);
writetable(NormalityTests, fullfile(tables_path, 'garch_normality_tests.csv'));

desc_stats = [mean(z_t); std(z_t); skewness(z_t); kurtosis(z_t); kurtosis(z_t)-3];
desc_labels = {'Mean'; 'Std Dev'; 'Skewness'; 'Kurtosis'; 'Excess Kurtosis'};
DescStats = table(desc_labels, desc_stats, 'VariableNames', {'Statistic', 'Value'});
writetable(DescStats, fullfile(tables_path, 'garch_descriptive_stats.csv'));

figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1);
histogram(z_t, 50, 'Normalization', 'pdf', 'FaceColor', 'b', 'EdgeColor', 'none');
hold on;
x_range = linspace(-5, 5, 100);
plot(x_range, normpdf(x_range, 0, 1), 'r-', 'LineWidth', 2);
xlabel('z_t');
ylabel('Density');
title('Standardized Residuals vs Normal');
legend('z_t', 'N(0,1)', 'Location', 'best');
grid on;

subplot(1,3,2);
qqplot(z_t);
title('Q-Q Plot of z_t');
grid on;

subplot(1,3,3);
correlogram(z_t.^2, 20, 1);
title('ACF of z_t^2');
grid on;

sgtitle('Diagnostics for GARCH Standardized Residuals', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_path, 'garch_diagnostics.png'));


% ________________ ESTIMATION GJR-GARCH(1,1) MODELS _____________________ %

Mdl_gjr = gjr(p, q);
[EstMdl_gjr, EstParamCov_gjr, logL_gjr] = estimate(Mdl_gjr, data_diff(1:wind), 'Display', 'off');

alpha_gjr = EstMdl_gjr.ARCH{1};
gamma_gjr = EstMdl_gjr.Leverage{1};
beta_gjr = EstMdl_gjr.GARCH{1};
omega_gjr = EstMdl_gjr.Constant;

% Extract standard errors from parameter covariance matrix
se_gjr = sqrt(diag(EstParamCov_gjr));

% Calculate t-statistics
coef_values_gjr = [omega_gjr; alpha_gjr; gamma_gjr; beta_gjr];
t_stats_gjr = coef_values_gjr ./ se_gjr;

% Calculate p-values (two-tailed test)
p_values_gjr = 2 * (1 - normcdf(abs(t_stats_gjr)));

% Create comprehensive parameter table (not saved)
gjr_params = table({'omega'; 'alpha'; 'gamma'; 'beta'}, ...
                   coef_values_gjr, ...
                   se_gjr, ...
                   t_stats_gjr, ...
                   p_values_gjr, ...
                   'VariableNames', {'Parameter', 'Coefficient', 'Std_Error', 't_Statistic', 'p_Value'});

% Display the table
disp('GJR-GARCH Model Parameter Estimates:');
disp(gjr_params);

% Calculate conditional variance
x2 = data_diff(1:wind).^2;
T = length(x2);
I_neg = (data_diff(1:wind) < 0);
sigma2_gjr = zeros(T+1, 1);
sigma2_gjr(1) = omega_gjr / (1 - alpha_gjr - beta_gjr - gamma_gjr/2);

for i = 1:T
    sigma2_gjr(i+1) = omega_gjr + (alpha_gjr + gamma_gjr * I_neg(i)) * x2(i) + beta_gjr * sigma2_gjr(i);
end

figure;
plot(time(1:wind), sigma2_gjr(2:end), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time');
ylabel('Conditional Variance');
title('GJR-GARCH Conditional Variance');
saveas(gcf, fullfile(figures_path, 'gjr_garch_variance.png'));

% Normality Analysis
z_t_gjr = data_diff(1:wind) ./ sqrt(sigma2_gjr(2:end));
[h_jb, pValue_jb, jbStat] = jbtest(z_t_gjr);
[h_cvm, pValue_cvm, A2_stat, ~] = cvmtest_known_normal(z_t_gjr);
[h_sw, pValue_sw] = swtest(z_t_gjr);

TestNames = {'Jarque-Bera'; 'Cramér-von Mises'; 'Shapiro-Wilk'};
TestStatistic = [jbStat; A2_stat; NaN];
PValue = [pValue_jb; pValue_cvm; pValue_sw];
RejectH0 = logical([h_jb; h_cvm; h_sw]);
Conclusion = cell(3,1);
for i = 1:3
    if RejectH0(i)
        Conclusion{i} = 'Non-Normal';
    else
        Conclusion{i} = 'Normal';
    end
end

NormalityTests_gjr = table(TestNames, TestStatistic, PValue, RejectH0, Conclusion);
disp('GJR-GARCH Normality Tests:');
disp(NormalityTests_gjr);

% Descriptive statistics
desc_stats = [mean(z_t_gjr); std(z_t_gjr); skewness(z_t_gjr); kurtosis(z_t_gjr); kurtosis(z_t_gjr)-3];
desc_labels = {'Mean'; 'Std Dev'; 'Skewness'; 'Kurtosis'; 'Excess Kurtosis'};
DescStats_gjr = table(desc_labels, desc_stats, 'VariableNames', {'Statistic', 'Value'});
disp('GJR-GARCH Descriptive Statistics:');
disp(DescStats_gjr);

% Diagnostics plots
figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1);
histogram(z_t_gjr, 50, 'Normalization', 'pdf', 'FaceColor', 'b', 'EdgeColor', 'none');
hold on;
x_range = linspace(-5, 5, 100);
plot(x_range, normpdf(x_range, 0, 1), 'r-', 'LineWidth', 2);
xlabel('z_t');
ylabel('Density');
title('Standardized Residuals vs Normal');
legend('z_t', 'N(0,1)', 'Location', 'best');
grid on;

subplot(1,3,2);
qqplot(z_t_gjr);
title('Q-Q Plot of z_t');
grid on;

subplot(1,3,3);
correlogram(z_t_gjr.^2, 20, 1);
title('ACF of z_t^2');
grid on;

sgtitle('Diagnostics for GJR-GARCH Standardized Residuals', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_path, 'gjr_garch_diagnostics.png'));

%______________________ MODEL SELECTION TESTING _________________________ %
LR = 2 * (logL_gjr - logL);
chi_square_critical = chi2inv(0.95, 1);

% Model comparison with statistics
model_comparison = table({'GARCH'; 'GJR-GARCH'; 'LR Test'; 'Critical Value (5%)'}, ...
                         [logL; logL_gjr; LR; chi_square_critical], ...
                         'VariableNames', {'Model', 'Value'});

disp('Model Comparison:');
disp(model_comparison);

% Determine which model is preferred
if LR > chi_square_critical
    fprintf('\nLikelihood Ratio Test Result: Reject H0 at 5%% level\n');
    fprintf('Conclusion: GJR-GARCH is significantly better than GARCH\n');
else
    fprintf('\nLikelihood Ratio Test Result: Fail to reject H0 at 5%% level\n');
    fprintf('Conclusion: No significant improvement with GJR-GARCH over GARCH\n');
end

%% -------------------------------------------------------------------- %%
%                   SECTION 3: WEEK (3) ANALYSIS                         %
%  --------------------------------------------------------------------  %

T_full = length(data_diff);
x = data_diff;
x2 = x.^2;
wind = floor(T_full/3)*2;
wind_for = T_full - wind;

E_GJR = NaN(wind_for, 1);
sigma_f_gjr_1 = NaN(wind_for, 1);
sigma_f_gjr_5 = NaN(wind_for, 1);

x2_realized = x2(wind+1:end);
sigma_prev_gjr = sigma2_gjr(end);

alpha_lev = [0.01, 0.05];
VaR_gjr = NaN(wind_for, 2);
exceed = zeros(wind_for, 2);

for i = 1:wind_for
    t = wind + i - 1;
    
    % One-step ahead forecast
    I_neg = (x(t) < 0);
    sigma_f_gjr_1(i) = omega_gjr + (alpha_gjr + I_neg * gamma_gjr) * x2(t) + beta_gjr * sigma_prev_gjr;
    sigma_prev_gjr = sigma_f_gjr_1(i);
    
    % Five-step ahead forecast
    sigma_temp = sigma_f_gjr_1(i);
    for h = 2:5
        sigma_temp = omega_gjr + (alpha_gjr + 0.5*gamma_gjr + beta_gjr) * sigma_temp;
    end
    sigma_f_gjr_5(i) = sigma_temp;
    
    % Forecast error (one-step ahead)
    E_GJR(i) = (x2_realized(i) - sigma_f_gjr_1(i))^2;
    
    % VaR calculation (one-step ahead, LEFT tail)
    r_realized = data_diff(wind + i);
    for j = 1:2
        VaR_gjr(i,j) = -sqrt(sigma_f_gjr_1(i)) * norminv(1 - alpha_lev(j));
        
        if r_realized < VaR_gjr(i,j)
            exceed(i,j) = 1;
        end
    end
end

N_exceed = sum(exceed, 1);
exceed_rate = N_exceed / wind_for;
expected_rate = alpha_lev;

MSFE_GJR = mean(E_GJR);

forecast_results = table({'MSFE GJR-GARCH'; ...
                          'VaR Exceedances 1% (count)'; 'VaR Exceedances 5% (count)'; ...
                          'VaR Exceedance Rate 1%'; 'VaR Exceedance Rate 5%'; ...
                          'Expected Rate 1%'; 'Expected Rate 5%'}, ...
                         [MSFE_GJR; N_exceed(1); N_exceed(2); ...
                          exceed_rate(1); exceed_rate(2); ...
                          expected_rate(1); expected_rate(2)], ...
                         'VariableNames', {'Metric', 'Value'});

disp('GJR-GARCH Forecast Results:');
disp(forecast_results);

time_forecast = time(wind+1:T_full);

figure('Position', [100, 100, 1200, 800]);
subplot(2,1,1);
plot(time_forecast, sqrt(sigma_f_gjr_1), 'b-', 'LineWidth', 1.5);
hold on;
plot(time_forecast, sqrt(sigma_f_gjr_5), 'r--', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Volatility Forecast');
title('GJR-GARCH Volatility Forecasts');
legend('One-Step Ahead', 'Five-Step Ahead');
grid on;

subplot(2,1,2);
plot(time_forecast, data_diff(wind+1:end), 'k-', 'LineWidth', 0.5);
hold on;
plot(time_forecast, VaR_gjr(:,1), 'r--', 'LineWidth', 1.5);
plot(time_forecast, VaR_gjr(:,2), 'b--', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Returns / VaR');
title('Value at Risk: Out-of-Sample Performance');
legend('Returns', 'VaR 1%', 'VaR 5%', 'Location', 'best');
grid on;

saveas(gcf, fullfile(figures_path, 'gjr_volatility_forecasts.png'));

disp('Forecasts completed successfully!');




%% --------------------------------------------------------------------- %
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
%                           WEEKLY ANALYSIS                              %
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
%  --------------------------------------------------------------------  %
%% -------------------------------------------------------------------- %%
%                  SECTION 1: WEEK (1 and 2) ANALYSIS - WEEKLY           %
%  --------------------------------------------------------------------  %

file_path = "/Users/filipponardoni/Desktop/university/LMEC^2/2° Year/Financial Econometrics/Project";
tables_path_w = fullfile(file_path, "Tables/Weekly");
figures_path_w = fullfile(file_path, "Figures/Weekly");

% Create directories if they don't exist
if ~exist(tables_path_w, 'dir')
    mkdir(tables_path_w);
end
if ~exist(figures_path_w, 'dir')
    mkdir(figures_path_w);
end

% Data Download
data_raw_w = readtable(fullfile(file_path, "/Data/Exchange_rate_weekly.csv"));
data_w = table2array(data_raw_w(:,2));
time_w = datetime(data_raw_w{:,1}, 'InputFormat', 'yyyy-MM-dd');
T_w = size(data_w,1);

% Count NaN values in the weekly data
nan_count_w = sum(isnan(data_w));
disp(['Number of NaN values in weekly data: ', num2str(nan_count_w)]);



% Using function fillmissing using mov mean averaged at 5
data_w = fillmissing(data_w, 'movmean', 5);

% Define crisis periods
financial_crisis_start = datetime(2007, 12, 1);
financial_crisis_end   = datetime(2009, 6, 30);
covid_crisis_start     = datetime(2020, 3, 1);
covid_crisis_end       = datetime(2020, 12, 31);


figure('Position', [100, 100, 900, 500]);

% Add shaded regions for crises FIRST (background)
hold on;

% Financial Crisis (2007–2009) — light shadow green
fill([financial_crisis_start, financial_crisis_end, ...
      financial_crisis_end, financial_crisis_start], ...
     [0.5, 0.5, 2, 2], ...
     [0.70 0.85 0.70], ...   % light green
     'FaceAlpha', 0.30, 'EdgeColor', 'none', ...
     'DisplayName', 'Financial crisis (2007--2009)');

% COVID-19 Crisis (2020) — darker shadow green
fill([covid_crisis_start, covid_crisis_end, ...
      covid_crisis_end, covid_crisis_start], ...
     [0.5, 0.5, 2, 2], ...
     [0.45 0.70 0.45], ...   % darker green
     'FaceAlpha', 0.30, 'EdgeColor', 'none', ...
     'DisplayName', 'COVID-19 shock (2020)');

% Plot exchange rate on top — dark green
plot(time_w, data_w, 'Color', [0.10 0.45 0.25], ...
     'LineWidth', 1.5, 'DisplayName', 'EUR/USD level');

grid on;
xlabel('Time', 'FontSize', 11);
ylabel('Exchange Rate', 'FontSize', 11);
title('EUR/USD Exchange Rate with Crisis Periods (Weekly)', ...
      'FontSize', 13, 'FontWeight', 'bold');

legend('Location', 'best');
hold off;

saveas(gcf, fullfile(figures_path_w, 'exchange_rate_w.png'));



data_diff_w = diff(log(data_w))*100;

figure('Position', [100, 100, 900, 500]);

% Add shaded regions for crises FIRST
hold on;

% y-limits based on the series (same logic as your template)
y_lim = [min(data_diff_w)*1.2, max(data_diff_w)*1.2];

% Financial Crisis (2007–2009)
fill([financial_crisis_start, financial_crisis_end, ...
      financial_crisis_end, financial_crisis_start], ...
     [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
     [0.70 0.85 0.70], ...
     'FaceAlpha', 0.30, 'EdgeColor', 'none', ...
     'DisplayName', 'Financial crisis (2007--2009)');

% COVID-19 Crisis (2020)
fill([covid_crisis_start, covid_crisis_end, ...
      covid_crisis_end, covid_crisis_start], ...
     [y_lim(1), y_lim(1), y_lim(2), y_lim(2)], ...
     [0.45 0.70 0.45], ...
     'FaceAlpha', 0.30, 'EdgeColor', 'none', ...
     'DisplayName', 'COVID-19 shock (2020)');

% Plot log returns — dark green
plot(time_w(2:end), data_diff_w, ...
     'Color', [0.10 0.45 0.25], ...
     'LineWidth', 1, 'DisplayName', 'Log returns');

% Mean line — dashed darker green
mean_log_returns = mean(data_diff_w, 'omitnan');
yline(mean_log_returns, '--', ...
      'Color', [0.00 0.35 0.20], ...
      'LineWidth', 1.5, ...
      'Label', sprintf('Mean: %.3f%%', mean_log_returns), ...
      'LabelHorizontalAlignment', 'left', ...
      'DisplayName', 'Sample mean');

grid on;
xlabel('Time', 'FontSize', 11);
ylabel('Log Returns (%)', 'FontSize', 11);
title('EUR/USD Log Returns with Crisis Periods (Weekly)', ...
      'FontSize', 13, 'FontWeight', 'bold');

legend('Location', 'best');
ylim(y_lim);
hold off;

saveas(gcf, fullfile(figures_path_w, 'log_returns_w.png'));


%________________________ RANDOM WALK ANALYSIS _________________________ %

% Initialize results savings
rand_walk_test_exc_w  = NaN(5,3);
rand_walk_test_diff_w = NaN(4,3);

% Perform different unit root tests on exchange rates
% 1] Augmented Dickey-Fuller test (1) lag
[h_adf_w, pValue_adf_w, stat_adf_w, criticalValues_adf_w] = adftest(data_w, "Lags", 1);
rand_walk_test_exc_w(1,:) = [pValue_adf_w, stat_adf_w, criticalValues_adf_w];  

% 2] ADF with deterministic trend
[h_w, pValue_w, stat_w, cValue_w] = adftest(data_w, "Model", "TS");
rand_walk_test_exc_w(2,:) = [pValue_w, stat_w, cValue_w];                      

% 3] Kwiatkowski-Phillips-Schmidt-Shin test
[h_kpss_w, pValue_kpss_w, stat_kpss_w, criticalValues_kpss_w] = kpsstest(data_w);
rand_walk_test_exc_w(3,:) = [pValue_kpss_w, stat_kpss_w, criticalValues_kpss_w]; 

% 4] Augmented Dickey-Fuller test (5) lag
[h_adf_w, pValue_adf_w, stat_adf_w, criticalValues_adf_w] = adftest(data_w, "Lags", 5);
rand_walk_test_exc_w(4,:) = [pValue_adf_w, stat_adf_w, criticalValues_adf_w]; 

% 5] Variance Ratio Test
[h_w, pValue_w, stat_w, cValue_w] = vratiotest(data_w);
rand_walk_test_exc_w(5,:) = [pValue_w, stat_w, cValue_w];                        

% Perform different unit root tests on differences exchange rates
% 1] Augmented Dickey-Fuller test (1) lag
[h_adf_w, pValue_adf_w, stat_adf_w, criticalValues_adf_w] = adftest(data_diff_w, "Lags", 1);
rand_walk_test_diff_w(1,:) = [pValue_adf_w, stat_adf_w, criticalValues_adf_w]; 
% 2] Kwiatkowski-Phillips-Schmidt-Shin test
[h_kpss_w, pValue_kpss_w, stat_kpss_w, criticalValues_kpss_w] = kpsstest(data_diff_w);
rand_walk_test_diff_w(2,:) = [pValue_kpss_w, stat_kpss_w, criticalValues_kpss_w]; 

% 3] Augmented Dickey-Fuller test (5) lag
[h_adf_w, pValue_adf_w, stat_adf_w, criticalValues_adf_w] = adftest(data_diff_w, "Lags", 5);
rand_walk_test_diff_w(3,:) = [pValue_adf_w, stat_adf_w, criticalValues_adf_w]; 

% 4] Variance Ratio Test
[h_w, pValue_w, stat_w, cValue_w] = vratiotest(data_diff_w);
rand_walk_test_diff_w(4,:) = [pValue_w, stat_w, cValue_w];                        


row_n_exc_w = ["DF", "DF-Trend", "KPSS", "ADF(5)", "VAR-Ratio"];
row_n_w     = ["DF", "KPSS", "ADF(5)", "VAR-Ratio"];
var_n_w     = ["p-value", "Test Statistic", "Critical Value (5\%)"];

table_rand_walk_exc_w  = rows2vars(array2table(rand_walk_test_exc_w,  "RowNames", row_n_exc_w, "VariableNames", var_n_w));
table_rand_walk_diff_w = rows2vars(array2table(rand_walk_test_diff_w, "RowNames", row_n_w,     "VariableNames", var_n_w));

writetable(table_rand_walk_exc_w,  fullfile(tables_path_w, 'unit_root_tests_levels_w.csv'));
writetable(table_rand_walk_diff_w, fullfile(tables_path_w, 'unit_root_tests_diff_w.csv'));


%________________________ STATIONARITY ANALYSIS _________________________ %

% Correlogram function called
c_w = correlogram(data_w, 100, 1);
saveas(gcf, fullfile(figures_path_w, 'correlogram_levels_w.png'));


d_w = correlogram(data_diff_w, 100, 1);
saveas(gcf, fullfile(figures_path_w, 'correlogram_diff_w.png'));

d_w = correlogram(data_diff_w, 100, 1, "^2", "^2");
saveas(gcf, fullfile(figures_path_w, 'correlogram_diff_sqrt_w.png'));



% Short Long run dependence Poterba-Summers
beta_w = PS_function(data_w, 1, 30);
figure;
plot(beta_w, 'Color', [0, 0.6, 0.3], 'LineWidth', 1.5);
grid on;
xlabel('Lag');
ylabel('Beta');
title('Poterba-Summers Test (Weekly)');
saveas(gcf, fullfile(figures_path_w, 'poterba_summers_w.png'));




%%_________________________ MOMENTS ANALYSIS ___________________________ %

% Report the Mandelbrot analysis
var_analysis_w = mand_function(data_diff_w, 1);
[mu_w, sigma_w, sym_w, kurt_w] = moments(data_diff_w);

moments_table_w = table({'Mean'; 'Std Dev'; 'Skewness'; 'Kurtosis'}, ...
                      [mu_w; sigma_w; sym_w; kurt_w], ...
                      'VariableNames', {'Statistic', 'Value'});
writetable(moments_table_w, fullfile(tables_path_w, 'descriptive_stats_w.csv'));

%%________________________ NORMALITY ANALYSIS __________________________ %

z_data_w = zscore(data_diff_w);

figure;
hold on;
histogram(z_data_w, 100, "Normalization", "pdf", "FaceAlpha", 0.4, "FaceColor", [0, 0.6, 0.3]);

x_w = linspace(-4, 4, 1000);
y_w = normpdf(x_w, 0, 1);

plot(x_w, y_w, 'r-', 'LineWidth', 1.8);
xline(3, 'k--', 'LineWidth', 1.5);
xline(-3, 'k--', 'LineWidth', 1.5);

[f_w, xi_w] = ksdensity(z_data_w);
plot(xi_w, f_w, 'Color', [0, 0.6, 0.3], 'LineWidth', 2);

extreme_idx_w = abs(z_data_w) > 3;
extreme_count_right_w = sum(z_data_w > 3);
extreme_count_left_w = sum(z_data_w < -3);

plot(z_data_w(extreme_idx_w), zeros(sum(extreme_idx_w),1), 'ro', 'MarkerSize', 6, 'LineWidth', 1.2);

xlabel("Standardized Returns");
ylabel("Density");
title("Histogram of Standardized Returns with N(0,1) Overlay (Weekly)");
legend({"Empirical PDF", "N(0,1)", "z=±3", "", "Kernel density", "Extreme values"}, 'Location', 'best');
grid on;
hold off;
saveas(gcf, fullfile(figures_path_w, 'normality_histogram_w.png'));



% Jarque–Bera
[h_jb_w, p_jb_w, jbStat_w] = jbtest(z_data_w);

% Kolmogorov–Smirnov
[h_ks_w, p_ks_w, ksStat_w] = kstest(z_data_w);

% Shapiro–Wilk
[h_sw_w, p_sw_w] = swtest(z_data_w);

% Cramér–von Mises
[h_cvm_w, p_cvm_w, A2_stat_w, ~] = cvmtest_known_normal(z_data_w);

% Collect results in a unified table
normality_tests_table_w = table( ...
    {'Jarque-Bera'; 'Kolmogorov-Smirnov'; 'Shapiro-Wilk'; 'Cramér--von Mises'}, ...
    [jbStat_w; ksStat_w; NaN; A2_stat_w], ...
    [p_jb_w; p_ks_w; p_sw_w; p_cvm_w], ...
    [h_jb_w; h_ks_w; h_sw_w; h_cvm_w], ...
    'VariableNames', {'Test', 'Statistic', 'p_value', 'Reject_H0'} ...
);

% Save results
writetable(normality_tests_table_w, ...
    fullfile(tables_path_w, 'normality_tests_summary_w.csv'));


[mu, sigma, sym, kurt] = moments(data_diff_w)

%% -------------------------------------------------------------------- %%
%                   SECTION 2: WEEK (2) ANALYSIS - WEEKLY               %
%  --------------------------------------------------------------------  %

% ________________ ESTIMATION GARCH(P,Q) MODELS _________________________ %

wind_w = floor(length(data_diff_w)/3)*2;
P_w = 10;
Q_w = 10;

aic_pq_w = zeros(P_w, Q_w);
bic_pq_w = zeros(P_w, Q_w);

min_aic_w = 99999999;
index_aic_min_w = NaN(2, 1);

min_bic_w = 99999999;
index_bic_min_w = NaN(2, 1);

for p_w = 1:P_w
    for q_w = 1:Q_w
        try
            Mdl_w = garch(p_w, q_w);
            [EstMdl_w, EstParamCov_w, logL_w] = estimate(Mdl_w, data_diff_w(1:wind_w), 'Display', 'off');

            k_w = 1 + p_w + q_w;
            T_w = length(data_diff_w);

            AIC_w = -2*logL_w + 2*k_w;
            BIC_w = -2*logL_w + k_w*log(T_w);

            aic_pq_w(p_w, q_w) = AIC_w;
            bic_pq_w(p_w, q_w) = BIC_w;

            if aic_pq_w(p_w,q_w) < min_aic_w
                min_aic_w = aic_pq_w(p_w,q_w);
                index_aic_min_w = [p_w, q_w];
            end

            if bic_pq_w(p_w,q_w) < min_bic_w
                min_bic_w = bic_pq_w(p_w,q_w);
                index_bic_min_w = [p_w, q_w];
            end

        catch
            aic_pq_w(p_w, q_w) = NaN;
            bic_pq_w(p_w, q_w) = NaN;
        end
    end
end

% Save AIC and BIC matrices
writematrix(aic_pq_w, fullfile(tables_path_w, 'aic_matrix_w.csv'));
writematrix(bic_pq_w, fullfile(tables_path_w, 'bic_matrix_w.csv'));

figure;
surf(aic_pq_w);
colormap([linspace(1,0,256)', linspace(1,0.6,256)', linspace(1,0.3,256)']);
hold on;
xlabel('q (ARCH lags)');
ylabel('p (GARCH lags)');
zlabel('AIC');
title('AIC Surface for GARCH(p,q) - Weekly');

pA_w = index_aic_min_w(1);
qA_w = index_aic_min_w(2);
zA_w = aic_pq_w(pA_w, qA_w);
plot3(qA_w, pA_w, zA_w, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
saveas(gcf, fullfile(figures_path_w, 'aic_surface_w.png'));

figure;
surf(bic_pq_w);
colormap([linspace(1,0,256)', linspace(1,0.6,256)', linspace(1,0.3,256)']);
hold on;
xlabel('q (ARCH lags)');
ylabel('p (GARCH lags)');
zlabel('BIC');
title('BIC Surface for GARCH(p,q) - Weekly');

pB_w = index_bic_min_w(1);
qB_w = index_bic_min_w(2);
zB_w = bic_pq_w(pB_w, qB_w);
plot3(qB_w, pB_w, zB_w, 'r*', 'MarkerSize', 12, 'LineWidth', 2);
saveas(gcf, fullfile(figures_path_w, 'bic_surface_w.png'));

% Model selected
p_w = pB_w;
q_w = qB_w;
Mdl_w = garch(p_w, q_w);

% Estimation
[EstMdl_w, EstParamCov_w, logL_w] = estimate( ...
    Mdl_w, data_diff_w(1:wind_w), 'Display', 'off');

% Extract parameters
alpha_w = EstMdl_w.ARCH{1};
beta_w  = EstMdl_w.GARCH{1};
omega_w = EstMdl_w.Constant;

% Standard errors
se_w = sqrt(diag(EstParamCov_w));

% Coefficient vector (same order as covariance matrix)
coef_values_w = [omega_w; alpha_w; beta_w];

% t-statistics
t_stats_w = coef_values_w ./ se_w;

% p-values (asymptotic normal)
p_values_w = 2 * (1 - normcdf(abs(t_stats_w)));

% Results table
garch_params_w = table( ...
    {'omega'; 'alpha'; 'beta'}, ...
    coef_values_w, ...
    se_w, ...
    t_stats_w, ...
    p_values_w, ...
    'VariableNames', {'Parameter', 'Coefficient', 'Std_Error', 't_Statistic', 'p_Value'} ...
);

% Display
disp('GARCH Model Parameter Estimates (Weekly):');
disp(garch_params_w);

% Save table
writetable(garch_params_w, ...
    fullfile(tables_path_w, 'garch_parameters_w.csv'));
x2_w = data_diff_w(1:wind_w).^2;

sigma2_w = zeros(length(x2_w)+1, 1);
sigma2_w(1) = omega_w / (1 - alpha_w - beta_w);

T_w = length(x2_w);
for i = 1:T_w
    sigma2_w(i+1) = omega_w + alpha_w * x2_w(i) + beta_w * sigma2_w(i);
end

figure;
plot(time_w(1:wind_w), sigma2_w(2:end), 'Color', [0, 0.6, 0.3], 'LineWidth', 1.5);
grid on;
xlabel('Time');
ylabel('Conditional Variance');
title('GARCH(p,q) Conditional Variance (Weekly)');
saveas(gcf, fullfile(figures_path_w, 'garch_variance_w.png'));

% Normality Analysis of z_t
z_t_w = data_diff_w(1:wind_w) ./ sqrt(sigma2_w(2:end));

[h_jb_w, pValue_jb_w, jbStat_w] = jbtest(z_t_w);
[h_cvm_w, pValue_cvm_w, A2_stat_w, ~] = cvmtest_known_normal(z_t_w);
[h_sw_w, pValue_sw_w] = swtest(z_t_w);

TestNames_w = {'Jarque-Bera'; 'Cramér-von Mises'; 'Shapiro-Wilk'};
TestStatistic_w = [jbStat_w; A2_stat_w; NaN];
PValue_w = [pValue_jb_w; pValue_cvm_w; pValue_sw_w];
RejectH0_w = logical([h_jb_w; h_cvm_w; h_sw_w]);
Conclusion_w = cell(3,1);
for i = 1:3
    if RejectH0_w(i)
        Conclusion_w{i} = 'Non-Normal';
    else
        Conclusion_w{i} = 'Normal';
    end
end

NormalityTests_w = table(TestNames_w, TestStatistic_w, PValue_w, RejectH0_w, Conclusion_w);
writetable(NormalityTests_w, fullfile(tables_path_w, 'garch_normality_tests_w.csv'));

desc_stats_w = [mean(z_t_w); std(z_t_w); skewness(z_t_w); kurtosis(z_t_w); kurtosis(z_t_w)-3];
desc_labels_w = {'Mean'; 'Std Dev'; 'Skewness'; 'Kurtosis'; 'Excess Kurtosis'};
DescStats_w = table(desc_labels_w, desc_stats_w, 'VariableNames', {'Statistic', 'Value'});
writetable(DescStats_w, fullfile(tables_path_w, 'garch_descriptive_stats_w.csv'));

figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1);
histogram(z_t_w, 50, 'Normalization', 'pdf', 'FaceColor', [0, 0.6, 0.3], 'EdgeColor', 'none');
hold on;
x_range_w = linspace(-5, 5, 100);
plot(x_range_w, normpdf(x_range_w, 0, 1), 'r-', 'LineWidth', 2);
xlabel('z_t');
ylabel('Density');
title('Standardized Residuals vs Normal');
legend('z_t', 'N(0,1)', 'Location', 'best');
grid on;

subplot(1,3,2);
qqplot(z_t_w);
title('Q-Q Plot of z_t');
grid on;

subplot(1,3,3);
correlogram(z_t_w.^2, 20, 1);
title('ACF of z_t^2');
grid on;

sgtitle('Diagnostics for GARCH Standardized Residuals (Weekly)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_path_w, 'garch_diagnostics_w.png'));

% ________________ ESTIMATION GJR-GARCH(1,1) MODELS _____________________ %

Mdl_gjr_w = gjr(p_w, q_w);
[EstMdl_gjr_w, EstParamCov_gjr_w, logL_gjr_w] = estimate(Mdl_gjr_w, data_diff_w(1:wind_w), 'Display', 'off');

alpha_gjr_w = EstMdl_gjr_w.ARCH{1};
gamma_gjr_w = EstMdl_gjr_w.Leverage{1};
beta_gjr_w = EstMdl_gjr_w.GARCH{1};
omega_gjr_w = EstMdl_gjr_w.Constant;

% Coefficient vector (same order as covariance matrix)
coef_values_gjr_w = [omega_gjr_w; alpha_gjr_w; gamma_gjr_w; beta_gjr_w];

% Standard errors
se_gjr_w = sqrt(diag(EstParamCov_gjr_w));

% t-statistics
t_stats_gjr_w = coef_values_gjr_w ./ se_gjr_w;

% p-values (asymptotic normal)
p_values_gjr_w = 2 * (1 - normcdf(abs(t_stats_gjr_w)));

% Results table
gjr_params_w = table( ...
    {'omega'; 'alpha'; 'gamma'; 'beta'}, ...
    coef_values_gjr_w, ...
    se_gjr_w, ...
    t_stats_gjr_w, ...
    p_values_gjr_w, ...
    'VariableNames', {'Parameter', 'Coefficient', 'Std_Error', 't_Statistic', 'p_Value'} ...
);

% Display
disp('GJR–GARCH Model Parameter Estimates (Weekly):');
disp(gjr_params_w);

% Save table
writetable(gjr_params_w, ...
    fullfile(tables_path_w, 'gjr_garch_parameters_w.csv'));
x2_w = data_diff_w(1:wind_w).^2;
T_w = length(x2_w);

I_neg_w = (data_diff_w(1:wind_w) < 0);

sigma2_gjr_w = zeros(T_w+1, 1);
sigma2_gjr_w(1) = omega_gjr_w / (1 - alpha_gjr_w - beta_gjr_w - gamma_gjr_w/2);

for i = 1:T_w
    sigma2_gjr_w(i+1) = omega_gjr_w + (alpha_gjr_w + gamma_gjr_w * I_neg_w(i)) * x2_w(i) + beta_gjr_w * sigma2_gjr_w(i);
end

figure;
plot(time_w(1:wind_w), sigma2_gjr_w(2:end), 'Color', [0, 0.6, 0.3], 'LineWidth', 1.5);
grid on;
xlabel('Time');
ylabel('Conditional Variance');
title('GJR-GARCH Conditional Variance (Weekly)');
saveas(gcf, fullfile(figures_path_w, 'gjr_garch_variance_w.png'));

z_t_gjr_w = data_diff_w(1:wind_w) ./ sqrt(sigma2_gjr_w(2:end));

[h_jb_w, pValue_jb_w, jbStat_w] = jbtest(z_t_gjr_w);
[h_cvm_w, pValue_cvm_w, A2_stat_w, ~] = cvmtest_known_normal(z_t_gjr_w);
[h_sw_w, pValue_sw_w] = swtest(z_t_gjr_w);

TestNames_w = {'Jarque-Bera'; 'Cramér-von Mises'; 'Shapiro-Wilk'};
TestStatistic_w = [jbStat_w; A2_stat_w; NaN];
PValue_w = [pValue_jb_w; pValue_cvm_w; pValue_sw_w];
RejectH0_w = logical([h_jb_w; h_cvm_w; h_sw_w]);
Conclusion_w = cell(3,1);
for i = 1:3
    if RejectH0_w(i)
        Conclusion_w{i} = 'Non-Normal';
    else
        Conclusion_w{i} = 'Normal';
    end
end

NormalityTests_gjr_w = table(TestNames_w, TestStatistic_w, PValue_w, RejectH0_w, Conclusion_w);
writetable(NormalityTests_gjr_w, fullfile(tables_path_w, 'gjr_garch_normality_tests_w.csv'));

desc_stats_w = [mean(z_t_gjr_w); std(z_t_gjr_w); skewness(z_t_gjr_w); kurtosis(z_t_gjr_w); kurtosis(z_t_gjr_w)-3];
desc_labels_w = {'Mean'; 'Std Dev'; 'Skewness'; 'Kurtosis'; 'Excess Kurtosis'};
DescStats_gjr_w = table(desc_labels_w, desc_stats_w, 'VariableNames', {'Statistic', 'Value'});
writetable(DescStats_gjr_w, fullfile(tables_path_w, 'gjr_garch_descriptive_stats_w.csv'));

figure('Position', [100, 100, 1200, 400]);

subplot(1,3,1);
histogram(z_t_gjr_w, 50, 'Normalization', 'pdf', 'FaceColor', [0, 0.6, 0.3], 'EdgeColor', 'none');
hold on;
x_range_w = linspace(-5, 5, 100);
plot(x_range_w, normpdf(x_range_w, 0, 1), 'r-', 'LineWidth', 2);
xlabel('z_t');
ylabel('Density');
title('Standardized Residuals vs Normal');
legend('z_t', 'N(0,1)', 'Location', 'best');
grid on;

subplot(1,3,2);
qqplot(z_t_gjr_w);
title('Q-Q Plot of z_t');
grid on;

subplot(1,3,3);
correlogram(z_t_gjr_w.^2, 20, 1);
title('ACF of z_t^2');
grid on;

sgtitle('Diagnostics for GJR-GARCH Standardized Residuals (Weekly)', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, fullfile(figures_path_w, 'gjr_garch_diagnostics_w.png'));

%______________________ MODEL SELECTION TESTING _________________________ %

LR_w = 2 * (logL_gjr_w - logL_w);
chi_square_critical_w = chi2inv(0.95, 1);

model_comparison_w = table({'GARCH'; 'GJR-GARCH'; 'LR Test'; 'Critical Value (5%)'}, ...
                         [logL_w; logL_gjr_w; LR_w; chi_square_critical_w], ...
                         'VariableNames', {'Model', 'Value'});
writetable(model_comparison_w, fullfile(tables_path_w, 'model_comparison_w.csv'));

%% -------------------------------------------------------------------- %%
%                   SECTION 3: WEEK (3) ANALYSIS                         %
%  --------------------------------------------------------------------  %

T_full_w = length(data_diff_w);
x_w  = data_diff_w;
x2_w = x_w.^2;

wind_w     = floor(T_full_w/3)*2;
wind_for_w = T_full_w - wind_w;

E_GARCH_w         = NaN(wind_for_w, 1);
sigma_f_garch_1_w = NaN(wind_for_w, 1);
sigma_f_garch_5_w = NaN(wind_for_w, 1);

x2_realized_w = x2_w(wind_w+1:end);
sigma_prev_w  = sigma2_w(end);

alpha_lev_w = [0.01, 0.05];
VaR_garch_w = NaN(wind_for_w, 2);
exceed_w    = zeros(wind_for_w, 2);

for i = 1:wind_for_w
    t = wind_w + i - 1;

    % One-step ahead variance forecast
    sigma_f_garch_1_w(i) = omega_w + alpha_w * x2_w(t) + beta_w * sigma_prev_w;
    sigma_prev_w = sigma_f_garch_1_w(i);

    % Five-step ahead variance forecast
    sigma_temp = sigma_f_garch_1_w(i);
    for h = 2:5
        sigma_temp = omega_w + (alpha_w + beta_w) * sigma_temp;
    end
    sigma_f_garch_5_w(i) = sigma_temp;

    % One-step ahead MSFE component
    E_GARCH_w(i) = (x2_realized_w(i) - sigma_f_garch_1_w(i))^2;

    % One-step ahead VaR (left tail) and exceedances
    r_realized_w = x_w(wind_w + i);
    for j = 1:2
        VaR_garch_w(i,j) = -sqrt(sigma_f_garch_1_w(i)) * norminv(1 - alpha_lev_w(j));
        if r_realized_w < VaR_garch_w(i,j)
            exceed_w(i,j) = 1;
        end
    end
end

N_exceed_w      = sum(exceed_w, 1);
exceed_rate_w   = N_exceed_w / wind_for_w;
expected_rate_w = alpha_lev_w;

MSFE_GARCH_w = mean(E_GARCH_w);

forecast_results_w = table({'MSFE GARCH'; ...
                            'VaR Exceedances 1% (count)'; 'VaR Exceedances 5% (count)'; ...
                            'VaR Exceedance Rate 1%'; 'VaR Exceedance Rate 5%'; ...
                            'Expected Rate 1%'; 'Expected Rate 5%'}, ...
                           [MSFE_GARCH_w; N_exceed_w(1); N_exceed_w(2); ...
                            exceed_rate_w(1); exceed_rate_w(2); ...
                            expected_rate_w(1); expected_rate_w(2)], ...
                           'VariableNames', {'Metric', 'Value'});

disp('GARCH Forecast Results (Weekly):');
disp(forecast_results_w);

writetable(forecast_results_w, fullfile(tables_path_w, 'garch_forecast_results_w.csv'));

time_forecast_w = time_w(wind_w+1:T_full_w);

figure('Position', [100, 100, 1200, 800]);

subplot(2,1,1);
plot(time_forecast_w, sqrt(sigma_f_garch_1_w), ...
     'Color', [0.10 0.45 0.25], 'LineWidth', 1.5);
hold on;
plot(time_forecast_w, sqrt(sigma_f_garch_5_w), '--', ...
     'Color', [0.00 0.35 0.20], 'LineWidth', 1.5);
xlabel('Time');
ylabel('Volatility Forecast');
title('GARCH Volatility Forecasts (Weekly)');
legend('One-Step Ahead', 'Five-Step Ahead', 'Location', 'best');
grid on;

subplot(2,1,2);
plot(time_forecast_w, x_w(wind_w+1:end), ...
     'Color', [0.10 0.45 0.25], 'LineWidth', 0.7);
hold on;
plot(time_forecast_w, VaR_garch_w(:,1), '--', ...
     'Color', "blue", 'LineWidth', 1.5);
plot(time_forecast_w, VaR_garch_w(:,2), '--', ...
     'Color', "red", 'LineWidth', 1.5);
xlabel('Time');
ylabel('Returns / VaR');
title('Value at Risk: Out-of-Sample Performance (Weekly)');
legend('Returns', 'VaR 1\%', 'VaR 5\%', 'Location', 'best');
grid on;

saveas(gcf, fullfile(figures_path_w, 'garch_volatility_forecasts_w.png'));

disp('Weekly forecasts completed successfully!');








%% --------------------------------------------------------------------- %
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
%                        BOOTSTRAP EXERCISE                              %
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %
%  --------------------------------------------------------------------  %

T = 1000;

% parameters
alpha = 0.6;
w = 0.012;
x_0 = 1;
z_t = randn(T, 1);

x = NaN(T,1);
sigma_2 = NaN(T,1);

x(1) = x_0;
sigma_2(1) = w + alpha * x(1)^2;   

for t = 2:T
    sigma_2(t) = w + alpha * x(t-1)^2;
    x(t) = sqrt(sigma_2(t)) * z_t(t);
end

l = 1;
Mdl = garch(0,l);
[EstMdl,EstParamCov,logL,info] = estimate(Mdl, x, 'Display','off');

w_est = EstMdl.Constant;
a_est = cell2mat(EstMdl.ARCH);

se_w = sqrt(EstParamCov(1,1));
se_a = sqrt(EstParamCov(2,2));

fprintf('ARCH(%d) estimation results:\n', l)
fprintf('omega = %.4f (se = %.4f)\n', w_est, se_w)
fprintf('alpha_1 = %.4f (se = %.4f)\n', a_est, se_a)
fprintf('Log-likelihood = %.2f\n', logL)



%% _________________________ Bootstrap __________________________________ %
B = 999;

z = x ./ sqrt(sigma_2);
z_c = (z - mean(z,'omitnan'))/std(z);



alpha_b = NaN(B,1);
w_b     = NaN(B,1);

w_hat = w_est;
a_hat = a_est;

for b = 1:B

    sigma_2_b = NaN(T,1);
    x_b       = NaN(T,1);

    x_b(1) = x(1);
    sigma_2_b(1) = w_hat + a_hat * x_b(1)^2;   

    for t = 2:T
        sigma_2_b(t) = w_hat + a_hat * x_b(t-1)^2;
        sigma_2_b(t) = max(sigma_2_b(t), 1e-12);   

        j = randi(T);
        x_b(t) = sqrt(sigma_2_b(t)) * z_c(j);
    end

    if any(~isfinite(x_b))                      
        continue
    end

    EstB = estimate(garch(0,1), x_b, 'Display','off');
    w_b(b) = EstB.Constant;
    alpha_b(b) = cell2mat(EstB.ARCH);

    fprintf('Bootstrap %d / %d...\n', b, B);
end




% Grid
xg = linspace(min(alpha_b,[],'omitnan'), ...
              max(alpha_b,[],'omitnan'), 500);

% Finite-sample Gaussian CDF
mu_hat    = a_est;
sigma_hat = std(alpha_b,'omitnan');   % bootstrap SD
F_fs      = normcdf(xg, mu_hat, sigma_hat);

figure
ecdf(alpha_b)
hold on
plot(xg, F_fs, 'r','LineWidth',2)


grid on
xlabel('\alpha')
ylabel('F(\alpha)')
legend('ECDF (bootstrap)', ...
       'Finite-sample Gaussian CDF', ...
       'Location','best')
title('ECDF vs finite-sample Gaussian approximation')



%% ____________________ Bootstrap Statistic _____________________________ %

% H0 value
alpha_bar = 0.6;

% Observed LR
M1 = garch(0,1);
[Est1,~,logL1] = estimate(M1, x, 'Display','off');     % unrestricted

M0 = garch(0,1);
M0.ARCH = {alpha_bar};
[Est0,~,logL0] = estimate(M0, x, 'Display','off');     % restricted (alpha fixed)

LR_obs = -2*(logL0 - logL1);

% Bootstrap settings
B = 999;
LR_star = NaN(B,1);
z = x ./ sqrt(sigma_2);
z_c = (z - mean(z,'omitnan'))/std(z);

w_H0 = Est0.Constant;

for b = 1:B
    x_b = NaN(T,1);
    s2b = NaN(T,1);

    x_b(1) = x(1);
    s2b(1) = max(w_H0 + alpha_bar*x_b(1)^2, 1e-12);

    for t = 2:T
        s2b(t) = max(w_H0 + alpha_bar*x_b(t-1)^2, 1e-12);
        x_b(t) = sqrt(s2b(t)) * z_c(randi(T));
    end

    if any(~isfinite(x_b)), continue; end

    try
        % unrestricted on bootstrap sample
        [~,~,logL1_b] = estimate(garch(0,1), x_b, 'Display','off');

        % restricted on bootstrap sample (alpha fixed)
        M0b = garch(0,1);
        M0b.ARCH = {alpha_bar};
        [~,~,logL0_b] = estimate(M0b, x_b, 'Display','off');

        LR_star(b) = -2*(logL0_b - logL1_b);
    catch
        continue
    end

    fprintf('Bootstrap %d / %d...\n', b, B);
end

p_boot = mean(LR_star >= LR_obs, 'omitnan');


figure
histogram(LR_star, 100, 'Normalization','pdf')
hold on
xg = linspace(0, max(LR_star,[],'omitnan'), 500);
plot(xg, chi2pdf(xg,1), 'b-', 'LineWidth', 1.5)
xline(LR_obs,'r','LineWidth',2)
grid on
xlabel('LR')
ylabel('Density')
title('Bootstrap LR distribution vs \chi^2_1')

legend('Bootstrap LR', '\chi^2_1 density', 'Observed LR', 'Location','best')


%% -------------------------------------------------------------------- %%
%                           FUNCTIONS                                    %
%  --------------------------------------------------------------------  %


% Indicator function
function val = ind_f(x, cond, c)
    if nargin < 3
        c = 0;
    end

    if nargin < 2
        cond = "<";
    end


    switch cond

        case ">"
            if x > c
                val = 1;
            else
                val = 0;
            end


        case "<"
            if x < c
                val = 1;
            else
                val = 0;
            end            

    end

end


% Correlogram function
function acf_values = correlogram(data, max_lag, plot_flag, tr_x1, tr_x2)
    if nargin < 5
        tr_x2 = "NORM";
    end
    if nargin < 4
        tr_x1 = "NORM";
    end
    if nargin < 3
        plot_flag = 0;
    end
    if nargin < 2
        max_lag = 20;
    end
    
    T = size(data,1);
    
    % Transformation for x1
    switch tr_x1
        case "NORM"
            x1 = data;
        case "ABS"
            x1 = abs(data);
        case "^2"
            x1 = data.^2;
    end
    
    % Transformation for x2
    switch tr_x2
        case "NORM"
            x2 = data;
        case "ABS"
            x2 = abs(data);
        case "^2"
            x2 = data.^2;
    end
    
    mu1 = mean(x1);
    mu2 = mean(x2);
    var1 = sum((x1 - mu1).^2);
    
    acf_values = zeros(max_lag,1);
    for k = 1:max_lag
        num = sum((x1(1:T-k) - mu1) .* (x2(k+1:T) - mu2));
        acf_values(k) = num / var1;
    end
    
    if plot_flag == 1
        stem(1:max_lag, acf_values, ...
            'filled', ...
            'Color', [0.10 0.45 0.25], ...
            'MarkerFaceColor', [0.10 0.45 0.25], ...
            'MarkerEdgeColor', [0.10 0.45 0.25]);
        xlabel('Lag');
        ylabel('Autocorrelation');
        title('Correlogram');
        grid on;
        conf_interval = 1.96 / sqrt(T);
        yline(conf_interval, 'r--', 'LineWidth', 1.5);
        yline(-conf_interval, 'r--', 'LineWidth', 1.5);
    end
end



function [betas, covar, var_x] = PS_function(data, l_min, l_max, plot_flag)
    if nargin < 4
        plot_flag = 1;
    end
    
    T = length(data);
    n_iter = l_max - l_min + 1;
    
    betas = zeros(n_iter, 1);
    covar = zeros(n_iter, 1);
    var_x = zeros(n_iter, 1);
    
    for j = 1:n_iter
        k = l_min + j - 1;
        T_k = T - k;
        
        data_m_k = zeros(T_k, 1);
        data_p_k = zeros(T_k, 1);
        
        for t = 1:T_k
            data_m_k(t) = (data(t+k) - data(t)) / data(t);
            data_p_k(t) = (data(t+k) - data(t)) / data(t);
        end
        
        data_m_k_lagged = zeros(T_k, 1);
        for t = 1:T_k
            if t > k
                data_m_k_lagged(t) = (data(t) - data(t-k)) / data(t-k);
            else
                data_m_k_lagged(t) = NaN;
            end
        end
        
        valid_idx = ~isnan(data_m_k_lagged);
        data_m_k_valid = data_m_k_lagged(valid_idx);
        data_p_k_valid = data_p_k(valid_idx);
        
        if ~isempty(data_m_k_valid)
            cov_matrix = cov(data_m_k_valid, data_p_k_valid, 1);
            covar(j) = cov_matrix(1, 2);
            var_x(j) = var(data_m_k_valid, 1);
            
            X = [ones(length(data_m_k_valid), 1), data_m_k_valid];
            Y = data_p_k_valid;
            beta_coeffs = (X' * X) \ (X' * Y);
            betas(j) = beta_coeffs(2);
        else
            covar(j) = NaN;
            var_x(j) = NaN;
            betas(j) = NaN;
        end
    end
    
    if plot_flag == 1
        ks = l_min:l_max;
        
        figure('Position', [100, 100, 1000, 600]);
        
        subplot(2,1,1);
        plot(ks, betas, 'b-', 'LineWidth', 2);
        hold on;
        yline(0, 'r--', 'LineWidth', 1.5, 'Label', 'Zero Line');
        xlabel('Horizon (k periods)', 'FontSize', 11);
        ylabel('Beta Coefficient', 'FontSize', 11);
        title('Poterba-Summers Test: Regression Coefficient \beta_k', 'FontSize', 13, 'FontWeight', 'bold');
        grid on;
        legend('\beta_k (negative → mean reversion)', 'Location', 'best');
        hold off;
        
        subplot(2,1,2);
        hold on;
        plot(ks, covar, 'b-', 'LineWidth', 2, 'DisplayName', 'Covariance');
        plot(ks, var_x, 'r-', 'LineWidth', 2, 'DisplayName', 'Variance');
        xlabel('Horizon (k periods)', 'FontSize', 11);
        ylabel('Value', 'FontSize', 11);
        title('Covariance and Variance across Horizons', 'FontSize', 13, 'FontWeight', 'bold');
        legend('Location', 'best');
        grid on;
        hold off;
    end
end




% Loretan and Phillips function
function [alpha] = Hill_est(data, m, rx)
    if nargin < 3
        rx = "DX";
    end
    
    [t, n] = size(data);
    
    if n ~= 1
        error('Input "data" must be a column vector.');
    end
    if m >= t
        error('m must be smaller than the sample size.');
    end
    
    if strcmp(rx, "DX")

        data_sorted = sort(data, 'descend');
        m_data = data_sorted(1:m);
        tail_label = 'Right Tail (DX)';
    else

        data_sorted = sort(data, 'ascend');
        m_data = abs(data_sorted(1:m)); 
        m_data = sort(m_data, 'descend'); 
        tail_label = 'Left Tail (SX)';
    end
    

    m_actual = length(m_data);
    x = zeros(m_actual, 1);
    y = zeros(m_actual, 1);
    
    for i = 1:m_actual
        x(i) = log(i / m_actual);    
        y(i) = log(m_data(i));     
    end
    
    % Plot
    figure;
    plot(x, y, 'o', 'MarkerSize', 4);
    xlabel('log(i/m)');
    ylabel('log(X_{(i)})');
    title(['Hill Tail Index Estimation - ' tail_label]);
    grid on;
    

    Xreg = [ones(m_actual, 1), x];
    b = (Xreg' * Xreg) \ (Xreg' * y);
    

    alpha = -1 / b(2);
    

    hold on;
    y_fit = Xreg * b;
    plot(x, y_fit, 'r-', 'LineWidth', 1.5);
    legend('Data', sprintf('Fit: α = %.3f', alpha), 'Location', 'best');
    hold off;
end



function[mu, sigma, sym, kurt] = moments(data)

    T=length(data);
    mu    = (1/T)*(sum(data));
    sigma =  (1/(T-1)) * sum((data - mu).^2);
    std_dev = sqrt(sigma);

    sym = (1/T) * sum( ((data - mu)/std_dev).^3 );
    kurt = (1/T) * sum( ((data - mu)/std_dev).^4 );

end


function [sigma_tau] = mand_function(data, plot_flag, rol_wind)
    if nargin < 2
        plot_flag = 1;
    end
    if nargin < 3
        rol_wind = 0;
    end
    
    T = length(data);
    
    if rol_wind == 0
        tau_max = floor(T / 3);
        sigma_tau = zeros(tau_max, 1);
        
        for i = 2:tau_max
            window_data = data(1:i);
            sigma_tau(i) = var(window_data, 1);
        end
        
        sigma_tau(1) = NaN;
        x_label = 'Window Size';
        title_str = 'Variance Estimation - Expanding Windows';
        
    else
        window_size = rol_wind;
        
        if window_size >= T
            error('Rolling window size must be smaller than data length');
        end
        
        n_windows = T - window_size + 1;
        sigma_tau = zeros(n_windows, 1);
        
        for i = 1:n_windows
            window_data = data(i:i+window_size-1);
            sigma_tau(i) = var(window_data, 1);
        end
        
        x_label = 'Window Position';
        title_str = sprintf('Variance Estimation - Rolling Window (size=%d)', window_size);
    end

    c = 1.358; % 95% rememberrr


    
    if plot_flag == 1
        figure;
        hold on;
        
        t = (1:length(sigma_tau))' / length(sigma_tau);
        bounds = c * sqrt(t .* (1 - t));
        mean_val = mean(sigma_tau(~isnan(sigma_tau)));
        
        plot(1:length(sigma_tau), sigma_tau, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Variance');
        yline(mean_val, 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('Mean: %.2e', mean_val));
        plot(1:length(sigma_tau), mean_val + bounds, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('+c\\surd{t(1-t)}, c=%.2f', c));
        plot(1:length(sigma_tau), mean_val - bounds, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('-c\\surd{t(1-t)}, c=%.2f', c));
        
        xlabel(x_label);
        ylabel('Estimated Variance (\sigma^2)');
        title(title_str);
        grid on;
        legend('show', 'Location', 'best');
        hold off;
    end
end



function [H, pValue, W2, W2star, crit] = cvmtest_known_normal(Y, alpha)
% ------------------------------------------------------------------------
%
%   University of Bologna
%   Author       : Filippo Nardoni
%   Last Updated : 08/09/2025
%
% -------------------------------------------------------------------------
% CVMTEST_KNOWN_NORMAL
%
%   Purpose:
%       Performs the Cramér–von Mises (CvM) goodness-of-fit test for the null
%       hypothesis that the sample Y comes from a standard normal
%       distribution N(0,1), assuming mean and variance are known (no re-estimation).
%
%   Reference:
%       Stephens, M.A. (1974). "EDF Statistics for Goodness of Fit and
%       Some Comparisons". Journal of the American Statistical Association.
%       Case 0 modification for fully specified distributions.
%
%   Syntax:
%       [H, pValue, W2, W2star, crit] = cvmtest_known_normal(Y, alpha)
%
%   Inputs:
%       Y      - Sample data (vector).
%       alpha  - Significance level (default = 0.05).
%
%   Outputs:
%       H       - Test decision:
%                   0 = Do not reject H0 at significance level alpha
%                   1 = Reject H0
%       pValue  - Approximate p-value (Stephens upper tail approximation).
%       W2      - CvM statistic.
%       W2star  - Stephens Case 0 modified statistic.
%       crit    - Asymptotic critical value for given alpha.
%
%   Notes:
%       - This version assumes H0: Y ~ N(0,1) with known parameters.
%       - Uses asymptotic critical values (Stephens 1974, Case 0).
% -------------------------------------------------------------------------

    if nargin < 2, alpha = 0.05; end
    Y = Y(:); Y = Y(~isnan(Y));
    n = numel(Y);
    Yord = sort(Y);

    % F under H0 (fully specified N(0,1))
    F = normcdf(Yord, 0, 1);

    % CvM statistic W^2
    i = (1:n)'; 
    W2 = (1/(12*n)) + sum( (F - (2*i-1)/(2*n)).^2 );

    % Stephens Case 0 modified statistic:
    W2star = (W2 - 0.4/n + 0.6/n^2) * (1 + 1/n);

    % Case 0 asymptotic critical values (Stephens 1974)
    A  = [0.10, 0.05, 0.01];           % significance levels
    CV = [0.347, 0.461, 0.743];        % corresponding critical values
    crit = interp1(A, CV, alpha, 'linear', 'extrap');

    % Decision rule
    H = (W2star > crit);

    % Approximate p-value (Stephens exponential tail approximation)
    z = (W2 - 0.4/n + 0.6/n^2) * (1 + 1/n);
    pValue = 0.05 * exp(2.79 - 6*z);
    pValue = max(min(pValue,1),0);

end

function [h, pValue, A2, critVals] = adtest_norm(Y, alpha)
% ------------------------------------------------------------------------
%
%   University of Bologna
%   Author       : Filippo Nardoni
%   Last Updated : 08/09/2025
%
% -------------------------------------------------------------------------
%   ADTEST_NORM  Anderson–Darling test for N(0,1) with known parameters.
%
%   [H, P] = ADTEST_NORM(Y) performs the Anderson–Darling test of the null 
%   hypothesis that the data in Y are i.i.d. N(0,1), assuming mean and 
%   variance are known (no re-estimation). 
%
%   INPUT:
%       Y     : sample data (vector)
%       alpha : significance level (default = 0.05)
%
%   OUTPUT:
%       h        : test decision (1 = reject H0, 0 = fail to reject)
%       pValue   : approximate p-value (Marsaglia & Marsaglia critical values)
%       A2       : Anderson–Darling test statistic
%       critVals : vector of critical values at [15%, 10%, 5%, 2.5%, 1%]
%
%   Reference:
%       Marsaglia, G. & Marsaglia, J.C. (2004). "Evaluating the Anderson–
%       Darling Distribution". Journal of Statistical Software.

    if nargin < 2
        alpha = 0.05;
    end

    % ------------------------------
    % 1. Sort sample and compute CDF
    % ------------------------------
    Y = sort(Y(:)); 
    n = length(Y);
    F = normcdf(Y,0,1);   % CDF under H0

    % avoid log(0)
    F(F==0) = eps;
    F(F==1) = 1 - eps;

    % ------------------------------
    % 2. Compute Anderson–Darling statistic
    % ------------------------------
    i = (1:n)';
    A2 = -n - (1/n) * sum( (2*i - 1) .* ( log(F) + log(1 - flipud(F)) ) );

    % ------------------------------
    % 3. Marsaglia & Marsaglia critical values (Case 0: known parameters)
    % ------------------------------
    alpha_levels = [0.15 0.10 0.05 0.025 0.01];
    critVals = [1.621 1.933 2.492 3.070 3.878];

    % ------------------------------
    % 4. Approximate p-value by interpolation
    % ------------------------------
    if A2 < critVals(1)
        pValue = 0.15;  % > 15%
    elseif A2 > critVals(end)
        pValue = 0.01;  % < 1%
    else
        idx = find(A2 < critVals, 1);
        x1 = critVals(idx-1); x2 = critVals(idx);
        y1 = alpha_levels(idx-1); y2 = alpha_levels(idx);
        pValue = interp1([x1,x2],[y1,y2],A2);
    end

    % ------------------------------
    % 5. Decision rule
    % ------------------------------
    critAlpha = interp1(alpha_levels,critVals,alpha,'linear','extrap');
    h = (A2 > critAlpha);

end


function [H, pValue] = swtest(x, alpha)

%SWTEST Shapiro-Wilk parametric hypothesis test of composite normality.
%   [H, pValue, SWstatistic] = SWTEST(X, ALPHA) performs the
%   Shapiro-Wilk test to determine if the null hypothesis of
%   composite normality is a reasonable assumption regarding the
%   population distribution of a random sample X. The desired significance 
%   level, ALPHA, is an optional scalar input (default = 0.05).
%
%   The Shapiro-Wilk and Shapiro-Francia null hypothesis is: 
%   "X is normal with unspecified mean and variance."
%
%   This is an omnibus test, and is generally considered relatively
%   powerful against a variety of alternatives.
%   Shapiro-Wilk test is better than the Shapiro-Francia test for
%   Platykurtic sample. Conversely, Shapiro-Francia test is better than the
%   Shapiro-Wilk test for Leptokurtic samples.
%
%   When the series 'X' is Leptokurtic, SWTEST performs the Shapiro-Francia
%   test, else (series 'X' is Platykurtic) SWTEST performs the
%   Shapiro-Wilk test.
% 
%    [H, pValue, SWstatistic] = SWTEST(X, ALPHA)
%
% Inputs:
%   X - a vector of deviates from an unknown distribution. The observation
%     number must exceed 3 and less than 5000.
%
% Optional inputs:
%   ALPHA - The significance level for the test (default = 0.05).
%  
% Outputs:
%  SWstatistic - The test statistic (non normalized).
%
%   pValue - is the p-value, or the probability of observing the given
%     result by chance given that the null hypothesis is true. Small values
%     of pValue cast doubt on the validity of the null hypothesis.
%
%     H = 0 => Do not reject the null hypothesis at significance level ALPHA.
%     H = 1 => Reject the null hypothesis at significance level ALPHA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                Copyright (c) 17 March 2009 by Ahmed Ben Saïda          %
%                 Department of Finance, IHEC Sousse - Tunisia           %
%                       Email: ahmedbensaida@yahoo.com                   %
%                    $ Revision 3.0 $ Date: 18 Juin 2014 $               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%
% References:
%
% - Royston P. "Remark AS R94", Applied Statistics (1995), Vol. 44,
%   No. 4, pp. 547-551.
%   AS R94 -- calculates Shapiro-Wilk normality test and P-value
%   for sample sizes 3 <= n <= 5000. Handles censored or uncensored data.
%   Corrects AS 181, which was found to be inaccurate for n > 50.
%   Subroutine can be found at: http://lib.stat.cmu.edu/apstat/R94
%
% - Royston P. "A pocket-calculator algorithm for the Shapiro-Francia test
%   for non-normality: An application to medicine", Statistics in Medecine
%   (1993a), Vol. 12, pp. 181-184.
%
% - Royston P. "A Toolkit for Testing Non-Normality in Complete and
%   Censored Samples", Journal of the Royal Statistical Society Series D
%   (1993b), Vol. 42, No. 1, pp. 37-43.
%
% - Royston P. "Approximating the Shapiro-Wilk W-test for non-normality",
%   Statistics and Computing (1992), Vol. 2, pp. 117-119.
%
% - Royston P. "An Extension of Shapiro and Wilk's W Test for Normality
%   to Large Samples", Journal of the Royal Statistical Society Series C
%   (1982a), Vol. 31, No. 2, pp. 115-124.
%

%
% Ensure the sample data is a VECTOR.
%

if numel(x) == length(x)
    x  =  x(:);               % Ensure a column vector.
else
    error(' Input sample ''X'' must be a vector.');
end

%
% Remove missing observations indicated by NaN's and check sample size.
%

x  =  x(~isnan(x));

if length(x) < 3
   error(' Sample vector ''X'' must have at least 3 valid observations.');
end

if length(x) > 5000
    persistent sw_warned
    if isempty(sw_warned) || ~sw_warned
        warning('Shapiro-Wilk test might be inaccurate due to large sample size ( > 5000).');
        sw_warned = true;
    end
end

%
% Ensure the significance level, ALPHA, is a 
% scalar, and set default if necessary.
%

if (nargin >= 2) && ~isempty(alpha)
   if ~isscalar(alpha)
      error(' Significance level ''Alpha'' must be a scalar.');
   end
   if (alpha <= 0 || alpha >= 1)
      error(' Significance level ''Alpha'' must be between 0 and 1.'); 
   end
else
   alpha  =  0.05;
end

% First, calculate the a's for weights as a function of the m's
% See Royston (1992, p. 117) and Royston (1993b, p. 38) for details
% in the approximation.

x       =   sort(x); % Sort the vector X in ascending order.
n       =   length(x);
mtilde  =   norminv(((1:n)' - 3/8) / (n + 1/4));
weights =   zeros(n,1); % Preallocate the weights.

if kurtosis(x) > 3
    
    % The Shapiro-Francia test is better for leptokurtic samples.
    
    weights =   1/sqrt(mtilde'*mtilde) * mtilde;

    %
    % The Shapiro-Francia statistic W' is calculated to avoid excessive
    % rounding errors for W' close to 1 (a potential problem in very
    % large samples).
    %

    W   =   (weights' * x)^2 / ((x - mean(x))' * (x - mean(x)));

    % Royston (1993a, p. 183):
    nu      =   log(n);
    u1      =   log(nu) - nu;
    u2      =   log(nu) + 2/nu;
    mu      =   -1.2725 + (1.0521 * u1);
    sigma   =   1.0308 - (0.26758 * u2);

    newSFstatistic  =   log(1 - W);

    %
    % Compute the normalized Shapiro-Francia statistic and its p-value.
    %

    NormalSFstatistic =   (newSFstatistic - mu) / sigma;
    
    % Computes the p-value, Royston (1993a, p. 183).
    pValue   =   1 - normcdf(NormalSFstatistic, 0, 1);
    
else
    
    % The Shapiro-Wilk test is better for platykurtic samples.

    c    =   1/sqrt(mtilde'*mtilde) * mtilde;
    u    =   1/sqrt(n);

    % Royston (1992, p. 117) and Royston (1993b, p. 38):
    PolyCoef_1   =   [-2.706056 , 4.434685 , -2.071190 , -0.147981 , 0.221157 , c(n)];
    PolyCoef_2   =   [-3.582633 , 5.682633 , -1.752461 , -0.293762 , 0.042981 , c(n-1)];

    % Royston (1992, p. 118) and Royston (1993b, p. 40, Table 1)
    PolyCoef_3   =   [-0.0006714 , 0.0250540 , -0.39978 , 0.54400];
    PolyCoef_4   =   [-0.0020322 , 0.0627670 , -0.77857 , 1.38220];
    PolyCoef_5   =   [0.00389150 , -0.083751 , -0.31082 , -1.5861];
    PolyCoef_6   =   [0.00303020 , -0.082676 , -0.48030];

    PolyCoef_7   =   [0.459 , -2.273];

    weights(n)   =   polyval(PolyCoef_1 , u);
    weights(1)   =   -weights(n);
    
    if n > 5
        weights(n-1) =   polyval(PolyCoef_2 , u);
        weights(2)   =   -weights(n-1);
    
        count  =   3;
        phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2 - 2 * mtilde(n-1)^2) / ...
                (1 - 2 * weights(n)^2 - 2 * weights(n-1)^2);
    else
        count  =   2;
        phi    =   (mtilde'*mtilde - 2 * mtilde(n)^2) / ...
                (1 - 2 * weights(n)^2);
    end
        
    % Special attention when n = 3 (this is a special case).
    if n == 3
        % Royston (1992, p. 117)
        weights(1)  =   1/sqrt(2);
        weights(n)  =   -weights(1);
        phi = 1;
    end

    %
    % The vector 'WEIGHTS' obtained next corresponds to the same coefficients
    % listed by Shapiro-Wilk in their original test for small samples.
    %

    weights(count : n-count+1)  =  mtilde(count : n-count+1) / sqrt(phi);

    %
    % The Shapiro-Wilk statistic W is calculated to avoid excessive rounding
    % errors for W close to 1 (a potential problem in very large samples).
    %

    W   =   (weights' * x) ^2 / ((x - mean(x))' * (x - mean(x)));

    %
    % Calculate the normalized W and its significance level (exact for
    % n = 3). Royston (1992, p. 118) and Royston (1993b, p. 40, Table 1).
    %

    newn    =   log(n);

    if (n >= 4) && (n <= 11)
    
        mu      =   polyval(PolyCoef_3 , n);
        sigma   =   exp(polyval(PolyCoef_4 , n));    
        gam     =   polyval(PolyCoef_7 , n);
    
        newSWstatistic  =   -log(gam-log(1-W));
    
    elseif n > 11
    
        mu      =   polyval(PolyCoef_5 , newn);
        sigma   =   exp(polyval(PolyCoef_6 , newn));
    
        newSWstatistic  =   log(1 - W);
    
    elseif n == 3
        mu      =   0;
        sigma   =   1;
        newSWstatistic  =   0;
    end

    %
    % Compute the normalized Shapiro-Wilk statistic and its p-value.
    %

    NormalSWstatistic   =   (newSWstatistic - mu) / sigma;
    
    % NormalSWstatistic is referred to the upper tail of N(0,1),
    % Royston (1992, p. 119).
    pValue       =   1 - normcdf(NormalSWstatistic, 0, 1);
    
    % Special attention when n = 3 (this is a special case).
    if n == 3
        pValue  =   6/pi * (asin(sqrt(W)) - asin(sqrt(3/4)));
        % Royston (1982a, p. 121)
    end
    
end

%
% To maintain consistency with existing Statistics Toolbox hypothesis
% tests, returning 'H = 0' implies that we 'Do not reject the null 
% hypothesis at the significance level of alpha' and 'H = 1' implies 
% that we 'Reject the null hypothesis at significance level of alpha.'
%

H  = (alpha >= pValue);


end

function [JBStat, pValue] = JarqueBeraTest(X)
    % X is a vector of sample data
    n = length(X); % sample size

    % Step 1: Calculate sample skewness and sample excess kurtosis
    S = skewness(X);
    K = kurtosis(X) - 3; % excess kurtosis

    % Step 2: Compute the Jarque-Bera statistic
    JBStat = n*((S^2)/6 + (K^2)/24);

    % Step 3: Compute the p-value from the chi-square distribution
    pValue = 1 - chi2cdf(JBStat,2);
end



function writeLatexTableScientific(T, filename, caption, label, varargin)
    % WRITELATEXTABLESCIENTIFIC Write MATLAB table to LaTeX format with booktabs
    %
    % Syntax:
    %   writeLatexTableScientific(T, filename, caption, label)
    %   writeLatexTableScientific(..., 'Name', Value)
    %
    % Inputs:
    %   T        - MATLAB table with RowNames and VariableNames
    %   filename - Full path to .tex file
    %   caption  - LaTeX caption string
    %   label    - LaTeX label (e.g., 'tab:unitroot_levels')
    %
    % Optional Name-Value Pairs:
    %   'Decimals'     - Number of decimal places (default: 4)
    %   'Position'     - Table position specifier (default: 'ht')
    %   'ColumnAlign'  - Cell array of column alignments (default: auto)
    %   'ScientificNotation' - Use scientific notation for small/large values (default: false)
    
    % Parse optional inputs
    p = inputParser;
    addParameter(p, 'Decimals', 4, @isnumeric);
    addParameter(p, 'Position', 'ht', @ischar);
    addParameter(p, 'ColumnAlign', {}, @iscell);
    addParameter(p, 'ScientificNotation', false, @islogical);
    parse(p, varargin{:});
    
    decimals = p.Results.Decimals;
    position = p.Results.Position;
    colAlign = p.Results.ColumnAlign;
    useScientific = p.Results.ScientificNotation;
    
    % Open file
    fid = fopen(filename, 'w');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    % Clean up on error
    cleanupObj = onCleanup(@() fclose(fid));
    
    % Table preamble
    fprintf(fid, '\\begin{table}[%s]\n', position);
    fprintf(fid, '\\centering\n');
    fprintf(fid, '\\caption{%s}\n', escapeLatexSpecialChars(caption));
    fprintf(fid, '\\label{%s}\n', label);
    
    % Column specification
    nCols = width(T);
    fprintf(fid, '\\begin{tabular}{l');
    
    if isempty(colAlign)
        % Default: S columns for numeric data
        for j = 1:nCols
            fprintf(fid, ' S');
        end
    else
        % User-specified alignment
        for j = 1:nCols
            fprintf(fid, ' %s', colAlign{j});
        end
    end
    
    fprintf(fid, '}\n');
    fprintf(fid, '\\toprule\n');
    
    % Header row
    fprintf(fid, 'Test');
    varNames = T.Properties.VariableNames;
    for j = 1:nCols
        fprintf(fid, ' & %s', escapeLatexSpecialChars(varNames{j}));
    end
    fprintf(fid, ' \\\\\n');
    fprintf(fid, '\\midrule\n');
    
    % Data rows
    rowNames = T.Properties.RowNames;
    for i = 1:height(T)
        fprintf(fid, '%s', escapeLatexSpecialChars(rowNames{i}));
        
        for j = 1:nCols
            val = T{i,j};
            
            if isnumeric(val) && ~isnan(val) && ~isinf(val)
                if useScientific && (abs(val) < 1e-3 || abs(val) > 1e4)
                    fprintf(fid, ' & %.4e', val);
                else
                    fprintf(fid, ' & %.*f', decimals, val);
                end
            elseif isnan(val)
                fprintf(fid, ' & --');
            elseif isinf(val)
                fprintf(fid, ' & $\\infty$');
            else
                % Non-numeric fallback
                fprintf(fid, ' & %s', escapeLatexSpecialChars(string(val)));
            end
        end
        
        fprintf(fid, ' \\\\\n');
    end
    
    % Table closing
    fprintf(fid, '\\bottomrule\n');
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table}\n');
    
    % File closed automatically by cleanupObj
end

function str = escapeLatexSpecialChars(str)
    % Escape special LaTeX characters
    str = char(str);
    str = strrep(str, '\', '\textbackslash{}');
    str = strrep(str, '&', '\&');
    str = strrep(str, '%', '\%');
    str = strrep(str, '$', '\$');
    str = strrep(str, '#', '\#');
    str = strrep(str, '_', '\_');
    str = strrep(str, '{', '\{');
    str = strrep(str, '}', '\}');
    str = strrep(str, '~', '\textasciitilde{}');
    str = strrep(str, '^', '\textasciicircum{}');
end



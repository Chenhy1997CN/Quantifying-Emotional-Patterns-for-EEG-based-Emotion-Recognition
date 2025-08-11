function [corr_p, p, index] = myttest2(D, N, threshold)
% myttest2: Perform two-sample t-test with FDR correction
% Inputs:
%   D: First data matrix [samples, features]
%   N: Second data matrix [samples, features]
%   threshold: FDR-corrected p-value threshold for significant features
% Outputs:
%   corr_p: FDR-corrected p-values
%   p: Raw p-values from t-test
%   index: Indices of features with corr_p < threshold

    % Perform two-sample t-test on each feature
    [~, p] = ttest2(D, N); % Ignore test statistics, keep p-values

    % Apply FDR correction using Benjamini-Hochberg method
    corr_p = mafdr(p, 'BHFDR', true);

    % Find indices of features with FDR-corrected p-values below threshold
    [~, index] = find(corr_p < threshold);
end
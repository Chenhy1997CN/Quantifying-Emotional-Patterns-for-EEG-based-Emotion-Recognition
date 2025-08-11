function [weight, corr_weight] = ttest2Weight(fea, lab)
% ttest2Weight: Compute feature weights using t-test and adjust by mean difference direction
% Inputs:
%   fea: Feature matrix [samples, features]
%   lab: Class labels (assumed to have exactly two classes)
% Outputs:
%   weight: Initial feature weights based on FDR-corrected p-values
%   corr_weight: Weights adjusted by mean difference direction between classes

    % Get unique class labels (assumes exactly two classes)
    classes = unique(lab);

    % Perform t-test with FDR correction (threshold = 0.05)
    [corr_p, ~] = myttest2(fea(lab == classes(1), :), fea(lab == classes(2), :), 0.05);

    % Set non-significant p-values (> 0.05) to 0.1
    corr_p(corr_p > 0.05) = 0.1;

    % Compute initial weights using double logarithm of inverse p-values
    weight = log10(log10(1 ./ corr_p));

    % Normalize weights to [0, 1] range
    weight = normalize(weight, 'range');

    % Find indices of significant features (weight > 0)
    index = find(weight > 0);

    % Initialize table to store mean values and differences
    tables = [];
    for i = 1:length(index)
        % Normalize feature values for the current significant feature
        f = normalize(fea(:, index(i)), 'range');
        % Compute mean for class 1
        tables(i, 1) = mean(f(lab == classes(1), :));
        % Compute mean for class 2
        tables(i, 2) = mean(f(lab == classes(2), :));
    end

    % Compute mean difference (class 1 - class 2)
    tables(:, 3) = tables(:, 1) - tables(:, 2);

    % Extract mean differences and normalize to direction (+1 or -1)
    p = tables(:, 3);
    p = p ./ abs(p);

    % Initialize corrected weights as initial weights
    corr_weight = weight;

    % Adjust weights of significant features by mean difference direction
    corr_weight(index) = corr_weight(index) .* p';
end
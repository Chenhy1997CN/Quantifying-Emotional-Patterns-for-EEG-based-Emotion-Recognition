function [rate, rates] = getSampleAggregationRate(fea, lab)
% getSampleAggregationRate: Compute sample aggregation rate using t-SNE and K-means
% Inputs:
%   fea: Feature matrix [samples, features]
%   lab: Class labels
% Outputs:
%   rate: Mean aggregation rate across classes
%   rates: Aggregation rates for each class based on t-SNE and K-means

    % Get unique class labels and number of classes
    classes = unique(lab);
    class_num = length(classes);

    % Set t-SNE perplexity as number of samples divided by number of classes
    Perplexity = size(fea, 1) / class_num;
    distance = 'correlation'; % Use correlation distance for t-SNE

    % Perform t-SNE dimensionality reduction
    Y = tsne(fea, 'Perplexity', Perplexity, 'Distance', distance);

    % Calculate mean coordinates for each class in t-SNE space
    for i = 1:class_num
        midPointsY(i, :) = mean(Y(lab == classes(i), :), 1);
    end

    % Perform K-means clustering on t-SNE results, initialized with class means
    idx = kmeans(Y, class_num, 'Start', midPointsY);

    % Compute aggregation rates using custom getCoverRate function
    [rates] = getCoverRate(Y, idx, class_num, Perplexity);

    % Compute mean aggregation rate
    rate = mean(rates);
end
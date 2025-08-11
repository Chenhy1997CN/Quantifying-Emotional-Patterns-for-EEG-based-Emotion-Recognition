function [kidx, k] = TrialCluster(data)
% TrialCluster: Cluster data using DBSCAN and K-means with t-SNE dimensionality reduction
% Input:
%   data: Feature matrix [samples, features]
% Outputs:
%   kidx: Final K-means cluster indices [samples, 1]
%   k: Estimated number of clusters

    % Get number of samples
    l = size(data, 1);

    % Set DBSCAN minimum points
    minpt = 8;

    % Set distance metric to correlation
    distance = 'correlation';

    % Normalize data using z-score
    data = normalize(data, 'zscore');

    % Compute pairwise correlation distances
    distmat = pdist(data, distance);

    % Sort distances and select top 10%
    distSeq = sort(distmat);
    distSeq = distSeq(1:round(0.1 * length(distSeq)));

    % Set exploration proportion based on distance sequence length
    if length(distSeq) > 1e4
        q = 0.1;
    else
        q = 0.5;
    end

    % Override q (hard-coded)
    q = 0.5;
    q = round(q * length(distSeq));

    % Initial DBSCAN clustering to estimate number of clusters
    for i = 1:q
        % Run DBSCAN with varying distance thresholds
        idx = dbscan(data, distSeq(i), minpt, 'Distance', distance);

        % Count noise points and effective clusters
        lossSample = sum(idx == -1);
        cluster_number = length(unique(idx));
        if lossSample > 0
            cluster_number = cluster_number - 1; % Exclude noise cluster
        end

        % Store cluster number and noise count
        cluster_numbers(i, 1) = cluster_number;
        lossSamples(i, 1) = lossSample;
    end

    % Create table of cluster numbers and noise counts
    table = [cluster_numbers, lossSamples];
    idxseq = table(:, 1);

    % Filter cluster numbers above threshold (40% of max)
    threshold = round(0.4 * max(idxseq));
    idxseq = idxseq(idxseq > threshold);

    % Find most frequent cluster number
    cluster_number = unique(idxseq);
    for i = 1:length(cluster_number)
        cluster_number(i, 2) = sum(idxseq == cluster_number(i, 1));
    end
    [~, i] = max(cluster_number(:, 2));
    cluster_number = cluster_number(i, 1);

    % Get minimum noise count for selected cluster number
    lossSample = table(table(:, 1) == cluster_number, 2);
    minloss = min(lossSample);

    % Select distance threshold for final DBSCAN
    i = table(:, 1) == cluster_number & table(:, 2) == minloss;
    elipse = mean(distSeq(i));

    % Run final DBSCAN
    idx = dbscan(data, elipse, minpt, 'Distance', distance);

    % Select seed points for K-means
    seed_index = [];
    for i = 1:cluster_number
        Index = find(idx == i);
        seed_index(i, 1) = Index(randperm(length(Index), 1));
    end
    k = cluster_number;
    guessk = k;

    % Iterative K-means clustering with t-SNE
    for i = 1:10
        % Perform t-SNE dimensionality reduction
        Y = tsne(data, 'Perplexity', round(l / guessk), 'Distance', distance);

        % Run K-means with seed points from t-SNE space
        kidx = kmeans(Y, cluster_number, 'Start', Y(seed_index, :));

        % Estimate new cluster number (assumes clusterGuess is defined)
        guessk = clusterGuess(kidx);

        % Update cluster number and seed points
        cluster_number = length(unique(kidx));
        seed_index = zeros(cluster_number, 1);
        for j = 1:cluster_number
            Index = find(kidx == j);
            seed_index(j, 1) = Index(randperm(length(Index), 1));
        end
    end
    k = guessk;

    % Final K-means clustering with fixed cluster number
    Y = tsne(data, 'Algorithm', 'exact', 'Perplexity', round(l / k), 'Distance', distance);
    kidx = kmeans(Y, k, 'Start', Y(seed_index, :));
end
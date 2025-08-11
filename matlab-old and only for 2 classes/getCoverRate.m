function [cover_rate] = getCoverRate(fea, idx, sumOfsubject, sumOftrial)
% getCoverRate: Compute aggregation rate for each subject based on K-means clustering
% Inputs:
%   fea: Feature matrix [samples, feature_dim] (e.g., raw or t-SNE reduced features)
%   idx: K-means cluster indices [samples, 1]
%   sumOfsubject: Number of subjects
%   sumOftrial: Number of trials per subject
% Output:
%   cover_rate: Aggregation rate for each subject

    % Initialize cover rate vector
    cover_rate = [];

    % Loop through each subject
    for i = 1:sumOfsubject
        % Extract cluster indices for the current subject
        class_vector = idx((i-1)*sumOftrial+1:i*sumOftrial, 1)';

        % Get unique cluster labels for the current subject
        classes = unique(class_vector);

        % Extract features for the current subject
        sub_f = fea((i-1)*sumOftrial+1:i*sumOftrial, :);

        % Count samples in each cluster
        counts = [];
        for j = 1:length(classes)
            counts(j) = sum(class_vector == classes(j));
        end

        % Compute cluster proportions and sort in descending order
        [rates, index] = sort(counts, 'descend');
        rates = rates / length(class_vector);

        % Initialize cluster distance flag
        coff = 0;

        % If more than one cluster exists, evaluate cluster distance
        if length(classes) > 1
            l = [];
            f = {};
            % Extract features for the top two clusters
            for k = 1:2
                f{k} = sub_f(class_vector == classes(index(k)), :);
                l = [l; repmat(k, [size(f{k}, 1), 1])];
            end
            f = [f{1}; f{2}];

            % Compute cluster distance (assumed to return distance metrics and a flag)
            [distance, cluster_distance, coff] = ClusterDistance(f, l);
        end

        % Compute cover rate: use top two clusters if coff == 1, else use top cluster
        if coff == 1
            cover_rate(i) = sum(rates(1:2));
        else
            cover_rate(i) = rates(1);
        end
    end

    % Return cover rate vector
    cover_rate = cover_rate;
end
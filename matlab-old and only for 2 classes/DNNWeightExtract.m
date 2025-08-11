function [weight] = DNNWeightExtract(f, label, varargin)
% DNNWeightExtract: Extract feature weights by iteratively training a DNN
% Inputs:
%   f: Feature matrix [samples, features]
%   label: Class labels
%   varargin: Optional argument for initial feature selection indices
% Output:
%   weight: Final feature weight vector

    % Initialize iteration counter
    iter = 1;

    % Handle feature selection index
    if nargin > 2
        index = zeros(1, size(f, 2)); % Initialize index vector with zeros
        index(varargin{1}) = 1; % Set specified indices to 1
    else
        index = ones(1, size(f, 2)); % Default: select all features
    end

    % Initialize variables
    acc = 100; % Initial accuracy to ensure loop entry
    w = []; % Matrix to store weights from each iteration
    weight = zeros(1, size(f, 2)); % Initialize final weight vector

    % Iterate while accuracy is above 90%
    while acc > 90
        % Train DNN and get weights and accuracy
        [w(iter, :), ~, acc] = DNNTrain(f, label, index);
        
        % Mark features with weights > 0.7 as processed
        index(w(iter, :) > 0.7) = -iter;
        
        % Assign weights > 0.7 to final weight vector
        weight(w(iter, :) > 0.7) = w(iter, w(iter, :) > 0.7);
        
        % Increment weights for features marked as processed
        weight(index < 0) = weight(index < 0) + 1;
        
        % Exit if sum of weights is zero
        if sum(w(iter, :)) == 0
            break;
        end
        
        % Increment iteration counter
        iter = iter + 1;
    end
end
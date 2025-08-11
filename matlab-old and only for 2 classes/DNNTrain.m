function [weight, weights, acc] = DNNTrain(featTrain, classTrain, varargin)
% DNNTrain: Train a Deep Neural Network for feature selection and classification
% Inputs:
%   featTrain: Training feature matrix [samples, features]
%   classTrain: Training class labels
%   varargin: Optional argument for feature selection index
% Outputs:
%   weight: Final feature weight vector
%   weights: Matrix of weights from multiple training iterations
%   acc: Mean training accuracy across iterations

    % Handle optional feature selection index
    if nargin == 3
        index = varargin{1}; % Use provided index for feature selection
    else
        index = ones(1, size(featTrain, 2)); % Default: select all features
    end
    featTrain = featTrain(:, index == 1); % Select features where index == 1

    % Get number of samples and features
    [samples, features] = size(featTrain);

    % Process class labels
    classes = unique(classTrain); % Get unique class labels
    numExtraClass = length(classes); % Number of unique classes
    for i = 1:numExtraClass
        % Re-encode class labels as integers from 1 to numExtraClass
        classTrain(classTrain == classes(i)) = i;
    end

    % Define neural network architecture
    layers = [ ...
        featureInputLayer(features) % Input layer for features
        fullyConnectedLayer(1, 'Weights', ones(1, features), 'Bias', 0) % First fully connected layer with initialized weights
        fullyConnectedLayer(numExtraClass) % Second fully connected layer for class outputs
        softmaxLayer % Softmax layer for probability distribution
        classificationLayer % Classification layer for loss computation
    ];

    % Configure training options
    options = trainingOptions('sgdm', ... % Use stochastic gradient descent with momentum
        'MaxEpochs', 20, ... % Maximum number of training epochs
        'MiniBatchSize', round(samples/10), ... % Mini-batch size (1/10 of samples)
        'InitialLearnRate', 0.1, ... % Initial learning rate
        'LearnRateSchedule', 'piecewise', ... % Learning rate decay schedule
        'LearnRateDropFactor', 0.2, ... % Reduce learning rate by factor of 0.2
        'LearnRateDropPeriod', 5, ... % Reduce learning rate every 5 epochs
        'ExecutionEnvironment', "cpu", ... % Use CPU for training
        'Plots', 'none', ... % Disable training progress plots
        'Verbose', false); % Suppress detailed training output

    % Train network multiple times
    for i = 1:10
        acc = 0; % Initialize accuracy
        iter = 1; % Initialize iteration counter
        % Train until accuracy reaches 90% or max 5 iterations
        while acc < 90
            [trainModel, info] = trainNetwork(featTrain, categorical(classTrain), layers, options); % Train network
            acc = info.TrainingAccuracy(end); % Get final training accuracy
            iter = iter + 1;
            if iter > 5
                break; % Exit if max iterations reached
            end
        end
        infos(i) = acc; % Store accuracy for this iteration

        % Extract activations and weights
        f = activations(trainModel, featTrain, 2)'; % Get activations from second layer (transposed)
        weight = trainModel.Layers(2,1); % Get weights of first fully connected layer
        % Decide whether to invert weights based on activation and class correlation
        if sum(~xor(f > mean(f), classTrain > mean(unique(classTrain)))) < samples/4
            weights(i,:) = -weight.Weights; % Invert weights if condition met
        else
            weights(i,:) = weight.Weights; % Use original weights
        end
    end

    % Compute mean accuracy across iterations
    acc = mean(infos);

    % Cluster weights using K-means
    l = kmeans(normalize(weights, 2, 'range'), 2); % Normalize weights and cluster into 2 groups
    [~, ll] = max([sum(l == 1), sum(l == 2)]); % Select cluster with more members
    w = mean(normalize(weights(l == ll, :), 2, 'range')); % Compute mean of normalized weights in selected cluster

    % Generate final weight vector
    weight = zeros(1, size(index, 2)); % Initialize output weight vector
    iter = 1; % Initialize iterator for selected features
    for i = 1:length(index)
        if index(i) == 1
            weight(i) = w(iter); % Assign weight to selected feature
            iter = iter + 1;
        end
    end
end
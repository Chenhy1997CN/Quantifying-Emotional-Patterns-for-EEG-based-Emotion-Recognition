EEG-based Emotion Recognition: Interpretable Study on EEG Individual Differences
This repository contains the source code for the paper "Quantifying Emotional Patterns for EEG-based Emotion Recognition: An Interpretable Study on EEG Individual Differences" by Huayu Chen et al. The code implements the proposed Correction T-test (CT) weight extraction method and the Weight-based Channel-model Matrix Framework (WCMF) for EEG-based emotion recognition, addressing individual differences across datasets (SEED, SEED-IV, SEED-V, RCLS, and MPED).
Repository Structure

matlab-old/: Contains MATLAB scripts for an earlier version of the emotion recognition pipeline, designed specifically for binary classification (2 classes).
Code.py: The main Python implementation of the WCMF and related methods for multi-class emotion recognition.

MATLAB Code Limitations
The MATLAB scripts in the matlab-old/ folder are designed for binary classification tasks only (e.g., positive vs. negative emotions). These scripts are considered legacy code and have the following limitations:

Binary Classification Only: The scripts are tailored for two-class problems and do not support multi-class (e.g., positive, neutral, negative) scenarios.
Outdated Implementation: The MATLAB code does not include the latest WCMF framework or CT weight extraction method described in the paper.
Compatibility: The scripts are tested and guaranteed to run on MATLAB R2022a. Other versions may encounter compatibility issues due to differences in function implementations (e.g., tsne, kmeans, or dbscan). Users are recommended to use MATLAB R2022a for optimal performance.

Python Code Description
The Code.py file contains the main implementation of the WCMF framework and related methods for multi-class EEG-based emotion recognition. Below is a brief overview of the key classes and methods:
Classes

SessionDataset (Dataset)

Purpose: A custom PyTorch dataset class for handling EEG session data, supporting feature and label management, downsampling, and trial label computation.
Attributes:
fea: Feature tensor (CUDA, float64).
lab: Label tensor (CUDA, int64).
classes: Unique class labels.
session_num: Number of sessions (fixed to 1).
trial_num: Number of trials per session.
trial_label: Trial labels for samples.


Methods:
__init__(session, fea, lab, tn): Initializes the dataset with features, labels, and trial number.
__len__(): Returns the number of samples.
__getitem__(idx): Retrieves a sample, its label, and trial label at the given index.
get_trial_label(): Computes trial labels based on unique labels per trial.
zscore(): Applies Z-score normalization to features.
downSampling(trial_num): Downsamples the dataset by selecting trial_num samples per trial.




ChannelModel (nn.Module)

Purpose: A simple neural network for processing single-channel EEG data.
Attributes:
fc1, fc2, fc3: Fully connected layers.
leaky_relu: LeakyReLU activation function.


Methods:
__init__(num_classes, input_dim, hidden_1, hidden_2): Initializes the model with specified dimensions.
forward(source): Performs forward pass through the network.




SingleChannelModelMatrix (nn.Module)

Purpose: A matrix of channel models with weight-based fusion for multi-channel EEG data, implementing the WCMF framework.
Attributes:
CT_weight: Correction T-test weights (classes × features).
confidence_weight: Confidence weights for features × classes.
weight: Channel × class weights for fusion.
classifiers: ModuleList of ChannelModel instances for each channel.
num_channels, classes: Number of channels and classes.


Methods:
__init__(opt, weight): Initializes the model with options and optional initial weights.
forward(sources): Performs forward pass, combining weighted outputs from channel models using parallel processing.
set_weight(cw): Sets CT weights and computes confidence weights for fusion.
calculate_CT_weight(session, times, downsample): Computes averaged CT weights over multiple downsampled iterations.
update_weights(session): Updates CT weights based on session data.





Key Functions

myttest2(D, N):

Performs a t-test between two groups and applies FDR correction to p-values.
Input: Two group data arrays (D, N).
Output: FDR-corrected p-values.


ttest2Weight(fea, lab):

Computes feature weights for binary classification using t-tests, with directional correction based on mean differences.
Input: Features (fea), labels (lab).
Output: Initial weights and corrected weights.


ttest2WeightMultiClass(fea, lab, num_classes):

Extends t-test weights to multi-class scenarios via pairwise comparisons, computing weights and a normalized emotional credibility score (type).
Input: Features, labels, number of classes (default=3).
Output: Aggregated weights, pairwise corrected weights, and type score.


calculate_CT_weight(session, times, downsample):

Computes averaged CT weights over multiple downsampled iterations to mitigate sample size bias.
Input: Session dataset, number of iterations, downsample size.
Output: Averaged CT weights.


imbalance_CT_weight(fea, lab, trial_label, times, min_segment_factor):

Computes CT weights handling class imbalance by iterative downsampling to the smallest trial size.
Input: Features, labels, trial labels, iterations, segment reduction factor.
Output: Averaged CT weights.


trial_cluster(data, minpt, distance, perplexity_fraction, stride):

Performs DBSCAN clustering with adaptive epsilon, standardizing data and evaluating cluster stability.
Input: Data, minimum points, distance metric, perplexity fraction, stride.
Output: Cluster labels, number of clusters, seed indices.


TrialKmeans(data, n):

Performs K-means clustering on t-SNE reduced data (2D, correlation metric).
Input: Data, number of clusters.
Output: Cluster labels.



Example Usage
The if __name__ == "__main__": block in Code.py provides example usage with dummy data, demonstrating dataset creation, downsampling, CT weight calculation, clustering, and model inference. To run the code:

Ensure dependencies (torch, numpy, sklearn, scipy, matplotlib) are installed.
Run Code.py with Python 3.8+ on a system with CUDA support for GPU acceleration.
Modify the dummy data or replace with actual EEG datasets (e.g., SEED) for real experiments.

Requirements

MATLAB: Version R2022a (other versions may encounter compatibility issues).
Python: 3.8 or higher.
Python Libraries:
torch
numpy
scikit-learn
scipy
matplotlib


Hardware: CUDA-enabled GPU recommended for Python code.

Notes

The MATLAB code in matlab-old/ is provided for reference but is not recommended for new projects due to its limitations.
The Python code (Code.py) is the primary implementation, supporting multi-class emotion recognition and the full WCMF framework.
For cross-dataset experiments, ensure EEG data follows the specifications described in the paper (e.g., 62-channel ESI NeuroScan, 200Hz sampling, 4s segments).

Citation
If you use this code in your research, please cite the following paper:

Huayu Chen, Xiaowei Li, Xuexiao Shao, Huanhuan He, Junxiang Li, Jing Zhu, Shuting Sun, Bin Hu, "Quantifying Emotional Patterns for EEG-based Emotion Recognition: An Interpretable Study on EEG Individual Differences," IEEE Transactions, 2025.

License
This project is licensed under the MIT License. See the LICENSE file for details.

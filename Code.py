import os
import random
import warnings
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import accuracy_score
from sklearn import svm
from scipy.interpolate import interp1d
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform, correlation
import matplotlib.pyplot as plt


class SessionDataset(Dataset):
    """
    A custom dataset class for handling session data in EEG-based emotion recognition.

    This class manages features and labels for EEG sessions, supports downsampling,
    and computes trial labels. It is designed to work with tensor data on CUDA devices.

    Args:
        session (dict, optional): Dictionary containing 'fea' (features) and 'lab' (labels). Defaults to {'fea': [], 'lab': []}.
        fea (torch.Tensor, optional): Feature tensor. Defaults to None.
        lab (torch.Tensor, optional): Label tensor. Defaults to None.
        tn (int, optional): Number of trials per session. Defaults to 200.

    Attributes:
        fea (torch.Tensor): Features tensor on CUDA.
        lab (torch.Tensor): Labels tensor on CUDA.
        classes (torch.Tensor): Unique classes in labels.
        session_num (int): Number of sessions (fixed to 1).
        trial_num (int): Number of trials.
        trial_label (torch.Tensor): Computed trial labels.
    """
    def __init__(self, session={'fea': [], 'lab': []}, fea=None, lab=None, tn=200):
        self.trial_label = None
        if fea is None:
            self.fea = torch.from_numpy(session['fea']).to(torch.float64).cuda()
            self.lab = torch.from_numpy(session['lab']).to(torch.int64).cuda()
        else:
            self.fea = fea.to(torch.float64).cuda()
            self.lab = lab.to(torch.int64).cuda()
        self.classes = torch.unique(self.lab)
        self.session_num = 1
        self.trial_num = tn
        self.get_trial_label()

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.lab)

    def __getitem__(self, idx):
        """
        Retrieves a sample, its label, and trial label at the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (sample, label, trial_label)
        """
        sample = self.fea[idx]
        label = self.lab[idx]
        trial_label = self.trial_label[idx]
        return sample, label, trial_label

    def get_trial_label(self):
        """
        Computes and sets trial labels based on unique labels in trials.

        Returns:
            tuple: (trial_unique_labels, trial_labels_tensor)
        """
        trial_labels = []
        trial_label_flat = []
        trials = len(self.fea) // self.trial_num

        for i in range(trials):
            start_idx = i * self.trial_num
            trial_labels.append(torch.unique(self.lab[start_idx:start_idx + self.trial_num]))
            trial_label_flat.extend([i + 1] * self.trial_num)

        self.trial_label = torch.tensor(trial_label_flat).cuda()
        return torch.tensor(trial_labels).cuda(), self.trial_label

    def zscore(self):
        """
        Applies Z-score normalization to the features.
        """
        mean = torch.mean(self.fea, dim=0)
        var = torch.std(self.fea, dim=0)
        self.fea = (self.fea - mean) / var

    def downSampling(self, trial_num):
        """
        Downsamples the dataset by randomly selecting samples from each trial.

        Args:
            trial_num (int): Number of samples to select per trial.

        Returns:
            SessionDataset: New downsampled dataset instance.
        """
        selected_indices = []

        for start_index in range(0, len(self.fea), self.trial_num):
            end_index = start_index + self.trial_num
            selected_indices.extend(random.sample(range(start_index, end_index), trial_num))

        selected_indices = np.array(selected_indices)
        fea_downsampled = self.fea[selected_indices]
        lab_downsampled = self.lab[selected_indices]

        return SessionDataset(fea=fea_downsampled, lab=lab_downsampled, tn=trial_num)


def myttest2(D, N):
    """
    Performs a t-test between two groups and applies FDR correction to p-values.

    This function computes independent t-tests for each feature dimension between two groups
    and corrects the p-values using False Discovery Rate (FDR) to account for multiple comparisons.

    Args:
        D (np.ndarray): Data for group 1.
        N (np.ndarray): Data for group 2.

    Returns:
        np.ndarray: Corrected p-values.
    """
    _, p = ttest_ind(D.astype(np.longdouble), N.astype(np.longdouble))
    corrected_p_values = multipletests(p, method='fdr_bh')[1]
    return corrected_p_values


def ttest2Weight(fea, lab):
    """
    Computes weights based on t-test between two classes, with correction.

    This function calculates statistical weights for binary classification based on inter-group differences.
    It uses corrected p-values to derive weights, and further computes directional corrections
    based on mean differences to indicate positive or negative relationships.

    Args:
        fea (np.ndarray or torch.Tensor): Features.
        lab (np.ndarray or torch.Tensor): Labels (binary classes).

    Returns:
        tuple: (weight, corrected_weight)
    """
    if isinstance(fea, torch.Tensor):
        fea = fea.cpu().numpy()
        lab = lab.cpu().numpy()

    classes = np.unique(lab)
    group1 = fea[lab == classes[0]]
    group2 = fea[lab == classes[1]]
    corr_p = myttest2(group1, group2)

    corr_p[corr_p > 0.05] = 0.1  # Set insignificant p-values to a baseline (0.1) to avoid over-weighting noise
    weight = np.log10(np.log10(1.0 / corr_p))  # Double log to compress the scale and emphasize small p-values
    weight[weight == np.inf] = 3  # Cap infinite values (from p=0) at 3 to prevent overflow and normalize

    max_score = 3
    weight = weight / max_score  # Normalize weights to [0,1] range

    index = np.where(weight > 0)[0]  # Indices of features with positive weights (significant differences)
    tables = np.zeros((len(index), 3))  # Table to store means and differences for significant features

    for i in range(len(index)):
        data = fea[:, index[i]]
        group1 = data[lab == classes[0]]
        group2 = data[lab == classes[1]]
        tables[i, 0] = np.mean(group1)  # Mean of group 1
        tables[i, 1] = np.mean(group2)  # Mean of group 2

    tables[:, 2] = tables[:, 0] - tables[:, 1]  # Difference between means
    p = tables[:, 2] / np.abs(tables[:, 2])  # Sign of the difference (+1 or -1) to indicate direction

    corr_weight = weight.copy()
    corr_weight[index] = corr_weight[index] * p  # Apply directional correction to weights
    return weight, corr_weight


def ttest2WeightMultiClass(fea, lab, num_classes=3):
    """
    Extends t-test weights to multi-class by pairwise comparisons.

    For multi-class (e.g., positive, neutral, negative), computes weights pairwise.
    Supports extension to more classes but may require rewriting for >3 classes.
    The function aggregates pairwise weights and computes a 'type' score representing
    normalized emotional credibility across feature dimensions.

    Args:
        fea (np.ndarray or torch.Tensor): Features.
        lab (np.ndarray or torch.Tensor): Labels.
        num_classes (int, optional): Number of classes. Defaults to 3.

    Returns:
        tuple: (weight, corrweights, type)
               - weight: Aggregated significance weights.
               - corrweights: Pairwise corrected weights.
               - type: Normalized score matrix indicating emotional pattern types (credibility) for each feature,
                       where values closer to 1 indicate high positive credibility, 0 neutral, and closer to -1 negative.
                       This helps in compatible multi-emotion classification by transforming differences into credibility scores.
    """
    if isinstance(fea, torch.Tensor):
        fea = fea.clone().detach().cpu().numpy()
        lab = lab.clone().detach().cpu().numpy()

    classes = np.unique(lab)
    num_features = fea.shape[1]
    weight = np.zeros((num_classes, num_features))
    corrweights = np.zeros((num_classes, num_features))
    index = 0
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            if i + 1 in classes and j + 1 in classes:
                group_indices = (lab == i + 1)
                group_data = fea[group_indices]
                group_labels = lab[group_indices]

                other_indices = (lab == j + 1)
                other_data = fea[other_indices]
                other_labels = lab[other_indices]
                data = np.concatenate([group_data, other_data], axis=0)
                label = np.concatenate([group_labels, other_labels], axis=0)
                weight[index, :], corrweights[index, :] = ttest2Weight(data, label)
            index += 1

    weight = np.sum(weight, axis=0)  # Sum pairwise weights to get overall significance per feature

    # w matrix: Transformation matrix for 3-class (positive-neutral-negative) scenario.
    # Each row/column corresponds to a class pair: [pos-neu, pos-neg, neu-neg].
    # Values indicate relationships: 1 for positive relation, -1 for negative, 0 for irrelevant.
    # This matrix transforms pairwise differences into emotional credibility scores compatible with multi-class.
    # For more classes, this matrix would need to be expanded or redesigned.
    w = torch.tensor([[0, 1, -1], [1, 0, -1], [1, -1, 0]], dtype=torch.float)
    cw = torch.tensor(corrweights, dtype=torch.float).T
    score = torch.matmul(cw, w)  # Compute score by matrix multiplication: pairwise weights transformed by relations

    score_max = torch.max(score, dim=1)[0]
    score_min = torch.min(score, dim=1)[0]

    type = (score - score_min.unsqueeze(1)) / (score_max - score_min).unsqueeze(1)  # Normalize scores to [0,1] per row
    type[torch.isnan(type)] = 0  # Handle NaN (e.g., division by zero if max==min) by setting to 0 (neutral)
    weight = torch.tensor(weight).cuda()
    corrweights = torch.tensor(corrweights).cuda()
    type = type.detach().cuda()
    return weight, corrweights, type


def calculate_CT_weight(session: SessionDataset, times=50, downsample=30):
    """
    Calculates averaged Correction T-test (CT) weights over multiple downsampled iterations.

    Downsampling is applied to reduce the impact of sample size on p-value magnitudes,
    ensuring more stable and comparable statistical significance across iterations.
    This helps in mitigating bias from large sample sizes that could overly shrink p-values.

    Args:
        session (SessionDataset): The session dataset.
        downsample (int, optional): Downsample size per trial. Defaults to 30.

    Returns:
        torch.Tensor: Averaged CT weights.
    """
    corr_CT_weight = torch.zeros(3, session.fea.shape[1]).cuda()
    for i in range(times):
        down_session = session.downSampling(downsample)
        _, CT_weight, _ = ttest2WeightMultiClass(down_session.fea, down_session.lab)
        corr_CT_weight = corr_CT_weight + CT_weight
    cw = corr_CT_weight / times
    return cw


def imbalance_CT_weight(fea, lab, trial_label, times=50, min_segment_factor=0.5):
    """
    Computes CT weights handling class imbalance via iterative downsampling.

    Iteratively reduces segment size until stable computation, averaging over multiple runs.
    This function addresses imbalance by subsampling trials to the smallest trial size,
    retrying with smaller segments if warnings occur (e.g., due to numerical issues).

    Args:
        fea (torch.Tensor): Features.
        lab (torch.Tensor): Labels.
        trial_label (torch.Tensor): Trial labels.
        times (int, optional): Number of iterations for averaging. Defaults to 10.
        min_segment_factor (float, optional): Factor to reduce segment size on failure. Defaults to 0.5.

    Returns:
        torch.Tensor: Averaged CT weights.
    """
    counts = torch.tensor([len(torch.where(trial_label == i)[0]) for i in torch.unique(trial_label)])
    trial_segment = counts[torch.argmin(counts)].item()
    iter_segment = trial_segment
    CT_weight = None
    while True:
        try:
            if iter_segment < trial_segment:
                selected_indices = []
                for trial in torch.unique(trial_label):
                    trial_index = torch.where(trial_label == trial)[0].tolist()
                    selected_indices.append(torch.tensor(random.sample(trial_index, iter_segment)))
                selected_indices = torch.cat(selected_indices, dim=0)
                fea_downsampled = fea[selected_indices]
                lab_downsampled = lab[selected_indices]
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    _, CT_weight, _ = ttest2WeightMultiClass(fea_downsampled, lab_downsampled)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    _, CT_weight, _ = ttest2WeightMultiClass(fea, lab)
        except Warning:
            iter_segment = int(iter_segment * min_segment_factor)
            continue
        break

    weight = CT_weight
    if iter_segment < trial_segment:
        for i in range(times):
            selected_indices = []
            for trial in torch.unique(trial_label):
                trial_index = torch.where(trial_label == trial)[0].tolist()
                selected_indices.append(torch.tensor(random.sample(trial_index, iter_segment)))
            selected_indices = torch.cat(selected_indices, dim=0)
            fea_downsampled = fea[selected_indices]
            lab_downsampled = lab[selected_indices]
            _, CT_weight, _ = ttest2WeightMultiClass(fea_downsampled, lab_downsampled)
            weight = weight + CT_weight
        weight = weight / times
    return weight


def trial_cluster(data, minpt=8, distance='correlation', perplexity_fraction=0.1, stride=None):
    """
    Performs clustering on trial data using DBSCAN with adaptive epsilon.

    This function standardizes data, computes pairwise distances, and searches for optimal epsilon
    by evaluating cluster stability and noise levels across a range of distances.

    Args:
        data (np.ndarray): Input data.
        minpt (int, optional): Minimum points for DBSCAN. Defaults to 8.
        distance (str, optional): Distance metric. Defaults to 'correlation'.
        perplexity_fraction (float, optional): Fraction for distance sequence. Defaults to 0.1.
        stride (int, optional): Step for epsilon search. Defaults to None (adaptive).

    Returns:
        tuple: (labels, cluster_number, seed_index)
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    distmat = pdist(data, metric=distance)
    distSeq = np.sort(distmat)
    distSeq = distSeq[distSeq > 0]
    distSeq = distSeq[:int(perplexity_fraction * len(distSeq))]

    if len(distSeq) == 0:
        return np.zeros(len(data)), 1, [0]  # Handle empty case

    if stride is None:
        stride = max(1, int(len(distSeq) / 50))  # Adaptive stride, e.g., divide into ~50 steps

    q = round(0.8 * len(distSeq))

    start_index = 0
    end_index = 0
    ini_cluster_number = 0
    for i in range(0, q, stride):
        if i >= len(distSeq):
            break
        dbscan = DBSCAN(eps=distSeq[i], min_samples=minpt, metric=distance, n_jobs=6)
        idx = dbscan.fit_predict(data)
        lossSample = np.sum(idx == -1)
        cluster_number = len(np.unique(idx))

        if lossSample < 0.01 * len(data):  # Threshold for acceptable noise level (1% outliers)
            start_index = i
            ini_cluster_number = cluster_number
            break

    for i in range(q, start_index, -stride):
        if i < 0 or i >= len(distSeq):
            continue
        dbscan = DBSCAN(eps=distSeq[i], min_samples=minpt, metric=distance, n_jobs=6)
        idx = dbscan.fit_predict(data)
        lossSample = np.sum(idx == -1)
        cluster_number = len(np.unique(idx))

        if lossSample > 0:
            cluster_number -= 1  # Adjust for noise if present
        if cluster_number > 5:  # Stop when clusters stabilize above a minimum (5)
            end_index = i
            break

    if start_index >= end_index or end_index == 0:
        # If invalid range or no end_index set, use full distSeq or a subset
        distSeq_slice = distSeq
    else:
        distSeq_slice = distSeq[start_index:end_index]

    cluster_numbers = [ini_cluster_number]
    distSequence = []
    if len(distSeq_slice) > 0:
        distSequence = [distSeq_slice[0]]
    else:
        distSequence = [np.mean(distSeq) if len(distSeq) > 0 else 0]

    finer_stride = max(1, int(len(distSeq_slice) / 10))  # Adaptive finer stride, e.g., ~10 steps
    for i in range(0, len(distSeq_slice), finer_stride):
        if i >= len(distSeq_slice):
            break
        dbscan = DBSCAN(eps=distSeq_slice[i], min_samples=minpt, metric=distance, n_jobs=6)
        idx = dbscan.fit_predict(data)
        lossSample = np.sum(idx == -1)
        cluster_number = len(np.unique(idx))

        if lossSample > 0:
            cluster_number -= 1
        if cluster_number == 1 and cluster_number < cluster_numbers[-1]:
            break

        cluster_numbers.append(cluster_number)
        distSequence.append(distSeq_slice[i])

    idxseq = np.array(cluster_numbers)
    threshold = int(np.mean(idxseq))  # Mean as threshold for stable clusters
    distSequence = np.array(distSequence)
    distSequence = distSequence[idxseq > threshold]
    idxseq = idxseq[idxseq > threshold]

    if len(idxseq) == 0:
        cluster_number = 1
        elipse = np.mean(distSeq) if len(distSeq) > 0 else 0
    else:
        cluster_classes = np.unique(idxseq)
        cluster_sum = np.zeros(len(cluster_classes))
        for i in range(len(cluster_classes)):
            cluster_sum[i] = np.sum(idxseq == cluster_classes[i])

        i_max = np.argmax(cluster_sum)  # Select most frequent cluster number
        cluster_number = cluster_classes[i_max]

        elipse = np.mean(distSequence[idxseq == cluster_number]) if len(distSequence) > 0 else 0

    dbscan = DBSCAN(eps=elipse, min_samples=minpt, metric=distance)
    idx = dbscan.fit_predict(data)

    seed_index = []
    unique_clusters = np.unique(idx[idx >= 0])  # Ignore noise (-1)
    cluster_number = len(unique_clusters)
    for c in unique_clusters:
        Index = np.where(idx == c)[0]
        if len(Index) > 0:
            seed_index.append(Index[np.random.choice(len(Index), size=1)][0])
        else:
            seed_index.append(0)  # Fallback if empty

    return idx, cluster_number, seed_index


def TrialKmeans(data, n):
    """
    Performs K-means clustering on t-SNE reduced data.

    Data is standardized, reduced to 2D via t-SNE (using correlation metric),
    then clustered with K-means.

    Args:
        data (np.ndarray): Input data.
        n (int): Number of clusters.

    Returns:
        np.ndarray: Cluster labels.
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    tsne = TSNE(n_components=2, perplexity=int(len(data) / n / 3), metric='correlation')
    data_2d = tsne.fit_transform(data)

    kmeans = KMeans(n_clusters=n, n_init=100)
    labels = kmeans.fit_predict(data_2d)

    return labels


class ChannelModel(nn.Module):
    """
    A simple neural network model for a single channel.

    Consists of three fully connected layers with LeakyReLU activations
    for processing single-channel EEG data.

    Args:
        num_classes (int): Number of output classes.
        input_dim (int): Input dimension.
        hidden_1 (int): Size of first hidden layer.
        hidden_2 (int): Size of second hidden layer.
    """
    def __init__(self, num_classes: int, input_dim, hidden_1, hidden_2):
        super(ChannelModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, num_classes)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, source):
        x = self.leaky_relu(self.fc1(source))
        x = self.leaky_relu(self.fc2(x))
        source_log_lossits = self.fc3(x)
        return source_log_lossits


class SingleChannelModelMatrix(nn.Module):
    """
    A matrix of channel models with weight-based fusion for multi-channel data.

    Supports CT weight setting and updating for emotion recognition.
    Uses parallel processing for efficiency in forward passes.

    Args:
        opt (argparse.Namespace): Options including num_classes, input_dim, num_channels.
        weight (torch.Tensor, optional): Initial weights. Defaults to None.
    """
    def __init__(self, opt, weight=None):
        super(SingleChannelModelMatrix, self).__init__()

        self.CT_weight = nn.Parameter(torch.zeros(opt.num_classes, opt.input_dim, dtype=torch.float64),
                                      requires_grad=False)  # CT weights for classes x features
        self.confidence_weight = nn.Parameter(torch.zeros(opt.input_dim, opt.num_classes, dtype=torch.float64),
                                              requires_grad=False)  # Confidence weights for features x classes
        if weight is None:
            self.weight = nn.Parameter(torch.ones(opt.num_channels, opt.num_classes, dtype=torch.float64))
        else:
            self.weight = weight

        self.num_channels = opt.num_channels
        self.classes = opt.num_classes
        self.classifiers = nn.ModuleList(
            [ChannelModel(opt.num_classes, 5, 15, 10) for _ in range(opt.num_channels)])  # One model per channel

    def forward(self, sources):
        combined_outputs = torch.zeros(sources.shape[0], self.classes, device='cuda:0')
        sources = sources.view((sources.shape[0], 5, -1))  # Reshape to batch x features_per_channel x channels

        def process_channel(i):
            classifier_output = self.classifiers[i](sources[:, :, i])
            return classifier_output * self.weight[i, :]  # Weight channel output

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_channel, range(self.num_channels)))

        for result in results:
            if result is not None:
                combined_outputs += result  # Sum weighted outputs across channels

        return combined_outputs


    def set_weight(self, cw):
        self.CT_weight = nn.Parameter(self.CT_weight + cw, requires_grad=False)
        # w matrix: Similar to ttest2WeightMultiClass, transforms absolute CT weights into confidence scores.
        # For 3 classes, it encodes pairwise relations to derive per-class confidence.
        # Note: This variant uses positive relations (1) for confidence summation, differing slightly from the signed version in ttest2WeightMultiClass.
        w = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.double).cuda()
        self.confidence_weight = nn.Parameter(torch.matmul(torch.abs(self.CT_weight.T), w).to(torch.float64),
                                              requires_grad=False)
        self.weight = nn.Parameter(
            torch.mean(self.confidence_weight.to(torch.float64).view(self.num_channels, -1, self.classes), dim=1),
            requires_grad=False)  # Average confidence into channel x class weights

    def calculate_CT_weight(self, session, times=50, downsample=30):
        corr_CT_weight = torch.zeros(3, session.fea.shape[1]).cuda()
        for i in range(times):
            down_session = session.downSampling(downsample)
            _, CT_weight, _ = ttest2WeightMultiClass(down_session.fea, down_session.lab)
            corr_CT_weight = corr_CT_weight + CT_weight
        cw = corr_CT_weight.T / times
        self.CT_weight = nn.Parameter(cw.to(torch.float64), requires_grad=False)
        w = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.double).cuda()
        self.confidence_weight = nn.Parameter(torch.matmul(torch.abs(cw), w).to(torch.float64),
                                              requires_grad=False)
        self.weight = nn.Parameter(
            torch.mean(self.confidence_weight.to(torch.float64).view(self.num_channels, -1, self.classes), dim=1),
            requires_grad=False)

    def update_weights(self, session):
        weight, corrweights, type = ttest2WeightMultiClass(session.fea, session.lab)
        self.CT_weight = self.CT_weight + corrweights
        cw = self.CT_weight.T
        w = torch.tensor([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.double).cuda()
        self.confidence_weight = torch.matmul(torch.abs(cw), w)
        self.weight = nn.Parameter(
            torch.mean(self.confidence_weight.to(torch.float64).view(self.num_channels, -1, self.classes), dim=1),
            requires_grad=False)

if __name__ == "__main__":
    # Example usage and testing
    # For testing, we'll create dummy data assuming 3 classes, 62 channels, 600 samples (3 trials of 200 each)
    dummy_fea = np.random.randn(600, 62 * 5)  # Assuming 5 features per channel, total input_dim=5*62=310
    dummy_lab = np.repeat([1, 2, 3], 200)  # Balanced labels for 3 classes

    # Create SessionDataset
    session = SessionDataset(fea=torch.tensor(dummy_fea), lab=torch.tensor(dummy_lab), tn=200)
    print(f"Dataset length: {len(session)}")
    print(f"Trial labels shape: {session.trial_label.shape}")

    # Test downsampling
    downsampled = session.downSampling(100)
    print(f"Downsampled length: {len(downsampled)}")

    # Test CT weight calculation
    ct_weight = calculate_CT_weight(session, downsample=100)
    print(f"CT weight shape: {ct_weight.shape}")

    # Test imbalance CT weight
    imb_weight = imbalance_CT_weight(session.fea, session.lab, session.trial_label, times=5)
    print(f"Imbalance CT weight shape: {imb_weight.shape}")

    # Test trial clustering (using a subset for speed)
    dummy_data = np.random.randn(100, 10)
    labels, num_clusters, seeds = trial_cluster(dummy_data, minpt=5)
    print(f"Cluster number: {num_clusters}")

    # Test TrialKmeans
    kmeans_labels = TrialKmeans(dummy_data, n=3)
    print(f"Kmeans labels shape: {kmeans_labels.shape}")

    # Test model
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=310)  # 62 channels * 5
    parser.add_argument('--num_channels', type=int, default=62)
    opt = parser.parse_args([])

    model = SingleChannelModelMatrix(opt)
    model = model.cuda()  # Move the model to CUDA to match the input device
    dummy_input = torch.randn(10, 310).cuda()  # Batch of 10
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

    # Update weights
    model.calculate_CT_weight(session, downsample=100)
    print("Weights updated successfully.")

    print("All tests passed!")
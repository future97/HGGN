import random

import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn import metrics
from sklearn.model_selection import KFold


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_score(label, output):
    auc = metrics.roc_auc_score(label, output)
    precision, recall, _ = metrics.precision_recall_curve(label, output)
    aupr = metrics.auc(recall, precision)
    pred = np.where(output > 0.5, 1, 0)

    accuracy = metrics.accuracy_score(label, pred)
    recall = metrics.recall_score(label, pred)
    precision = metrics.precision_score(label, pred)
    f1_score = metrics.f1_score(label, pred)
    return auc, aupr, accuracy, recall, precision, f1_score


def calculate_auc_aupr(output, label, mask, neg_mask):
    mask_sum = mask + neg_mask
    output = output.reshape(-1)[np.where(mask_sum.reshape(-1) == 1)]
    label = label.reshape(-1)[np.where(mask_sum.reshape(-1) == 1)]
    auc = metrics.roc_auc_score(label, output)
    precision, recall, _ = metrics.precision_recall_curve(label, output)
    aupr = metrics.auc(recall, precision)
    return auc, aupr


def load_data(data_dir, k_index):
    rna_features = np.load(data_dir + 'rna_features.npy')
    drug_features = np.load(data_dir + 'drug_features.npy')
    inter_features_rna = np.load(data_dir + 'inter_features_rna.npy')
    inter_features_drug = np.load(data_dir + 'inter_features_drug.npy')
    adj = np.load(data_dir + 'adj.npy')
    interaction = np.load(data_dir + 'interaction.npy')

    coo_inter = coo_matrix(interaction)
    pos_data = np.hstack((coo_inter.row[:, np.newaxis], coo_inter.col[:, np.newaxis]))
    neg_data = np.array(random.choices(np.vstack(np.where(interaction == 0)).transpose(), k=pos_data.shape[0]))
    skf = KFold(n_splits=5, shuffle=True)
    for fold_index, (train_index, test_index) in enumerate(skf.split(pos_data)):
        if fold_index == k_index:
            train_data, test_data = pos_data[train_index], pos_data[test_index]
            train_neg_data, test_neg_data = neg_data[train_index], neg_data[test_index]
    return adj, interaction, rna_features, drug_features, inter_features_rna, inter_features_drug, train_data, test_data, train_neg_data, test_neg_data

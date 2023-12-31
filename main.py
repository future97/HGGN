import argparse

import scipy.sparse as sp

from HGGN import *
from utils import *

seed = 10
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epoch):
    for i in range(epoch):
        model.zero_grad()
        model.train()
        x = model.projection_and_aggregation(rna_features, drug_features, inter_features_rna, inter_features_drug)
        train_output = model(x, adj, train_data)
        train_loss = loss_function(train_output, train_data_label)
        train_auc = metrics.roc_auc_score(train_data_label.detach().cpu().numpy(), train_output.detach().cpu().numpy())
        precision, recall, _ = metrics.precision_recall_curve(train_data_label.detach().cpu().numpy(),
                                                              train_output.detach().cpu().numpy())
        train_aupr = metrics.auc(recall, precision)
        print(
            f'Epoch:{i + 1} Train - Loss: {train_loss.detach().cpu().numpy()}, - AUC: {train_auc} - AUPR: {train_aupr}')
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            x = model.projection_and_aggregation(rna_features, drug_features, inter_features_rna, inter_features_drug)
            test_output = model(x, adj, test_data)
            test_loss = loss_function(test_output, test_data_label)
            test_auc, test_aupr, test_accuracy, test_recall, test_precision, test_f1_score = calculate_score(
                test_data_label.detach().cpu().numpy(), test_output.detach().cpu().numpy())
            print(f'Test - Loss: {test_loss.detach().cpu().numpy()}, - AUC: {test_auc} - AUPR: {test_aupr}')


def test():
    model.eval()
    with torch.no_grad():
        x = model.projection_and_aggregation(rna_features, drug_features, inter_features_rna, inter_features_drug)
        test_output = model(x, adj, test_data)
        test_loss = loss_function(test_output, test_data_label)
        test_auc, test_aupr, test_accuracy, test_recall, test_precision, test_f1_score = calculate_score(
            test_data_label.detach().cpu().numpy(), test_output.detach().cpu().numpy())
        print(f'Test - Loss: {test_loss.detach().cpu().numpy()}, - AUC: {test_auc} - AUPR: {test_aupr}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default='./data/')
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--hid-r", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-features", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    args = parser.parse_args()

    # load data
    adj, interaction, rna_features, drug_features, inter_features_rna, inter_features_drug, \
        train_pos_data, test_pos_data, train_neg_data, test_neg_data = load_data(data_dir=args.data_dir,
                                                                                 k_index=args.fold)
    global_node_num = int(adj.shape[0] * 0.1)
    adj = np.vstack((np.hstack((adj, np.ones(shape=(adj.shape[0], global_node_num)))),
                     np.hstack((np.ones(shape=(global_node_num, adj.shape[0])),
                                np.zeros((global_node_num, global_node_num))))))

    train_data = np.vstack([train_pos_data, train_neg_data])
    train_data_label = torch.tensor(
        np.vstack([np.ones([train_pos_data.shape[0], 1]), np.zeros([train_neg_data.shape[0], 1])]),
        dtype=torch.float32).to(device)
    test_data = np.vstack([test_pos_data, test_neg_data])
    test_data_label = torch.tensor(
        np.vstack([np.ones([test_pos_data.shape[0], 1]), np.zeros([test_neg_data.shape[0], 1])]),
        dtype=torch.float32).to(device)

    sp_adj = sp.coo_matrix(adj)
    indices = np.vstack((sp_adj.row, sp_adj.col))
    adj = torch.LongTensor(indices).to(device)
    interaction = torch.tensor(interaction).to(device)
    rna_features = torch.tensor(rna_features).to(device)
    drug_features = torch.tensor(drug_features).to(device)
    inter_features_rna = torch.tensor(inter_features_rna).to(device)
    inter_features_drug = torch.tensor(inter_features_drug).to(device)

    model = HGGN(r=args.hid_r, n_layers=args.n_layers, n_features=args.n_features,
                 num_rna=rna_features.shape[0],
                 num_dis=drug_features.shape[0],
                 n_global_node=global_node_num
                 )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.BCELoss()

    train(args.num_epochs)
    test()

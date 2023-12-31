import torch
import torch.nn as nn
import torch.nn.functional as F

from models import GATConv


class HGGN(nn.Module):
    def __init__(self, r, n_layers, n_features, num_rna, num_dis, n_global_node):
        super(HGGN, self).__init__()

        self.num_rna = num_rna
        self.num_dis = num_dis
        self.n_global_node = n_global_node
        n_node = num_rna + num_dis + n_global_node

        self.w_r_self = nn.Parameter(torch.zeros(size=(num_rna, n_features)))
        nn.init.xavier_uniform_(self.w_r_self.data)

        self.w_r_proj = nn.Parameter(torch.zeros(size=(num_rna, n_features)))
        nn.init.xavier_uniform_(self.w_r_proj.data)

        self.w_d_self = nn.Parameter(torch.zeros(size=(num_dis, n_features)))
        nn.init.xavier_uniform_(self.w_d_self.data)

        self.w_d_proj = nn.Parameter(torch.zeros(size=(num_dis, n_features)))
        nn.init.xavier_uniform_(self.w_d_proj.data)

        self.GATLayers = nn.ModuleList([
            GATConv(in_channels=n_features if i == 0 else r, out_channels=r) for i in range(n_layers)])

        self.MLP = nn.Sequential(nn.Linear(2 * r, r),
                                 nn.ReLU(),
                                 nn.Linear(r, int(r / 2)),
                                 nn.ReLU(),
                                 nn.Linear(int(r / 2), 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 16),
                                 nn.ReLU(),
                                 nn.Linear(16, 8),
                                 nn.ReLU(),
                                 torch.nn.Linear(8, 1),
                                 nn.Sigmoid())

    def projection_and_aggregation(self, rna_features, drug_features, inter_features_rna, inter_features_drug):
        # self-attention aggregation
        x_rna1 = torch.mm(inter_features_rna, self.w_r_proj)
        x_rna2 = torch.mm(rna_features, self.w_r_self)
        attention_weights1 = F.softmax(torch.matmul(x_rna1, x_rna2.T), dim=1)
        x1 = torch.matmul(attention_weights1.unsqueeze(1), x_rna2.unsqueeze(0)).squeeze(1)
        x_drug1 = torch.mm(inter_features_drug, self.w_d_proj)
        x_drug2 = torch.mm(drug_features, self.w_d_self)
        attention_weights2 = F.softmax(torch.matmul(x_drug1, x_drug2.T), dim=1)
        x2 = torch.matmul(attention_weights2.unsqueeze(1), x_drug2.unsqueeze(0)).squeeze(1)
        global_feature = torch.zeros(size=(self.n_global_node, x1.shape[1])).to(torch.device(x1.device))
        return torch.vstack([x1, x2, global_feature])

    def embedding(self, x, adj, coo_data):
        for GATLayer in self.GATLayers:
            x = GATLayer(x, adj)

        x_1 = x[:self.num_rna, :]
        x_2 = x[self.num_rna:(self.num_rna + self.num_dis), :]

        x_new = torch.vstack([torch.hstack([x_1[row[0]], x_2[row[1]]]) for row in coo_data])
        return x_new

    def forward(self, x, adj, coo_data):
        x_new = self.embedding(x, adj, coo_data)
        x_new = self.MLP(x_new)
        return x_new

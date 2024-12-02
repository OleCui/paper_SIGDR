import math
import torch
import torch.nn as nn
import dgl.nn.pytorch
from parse_args import args
import torch.nn.functional as F
from graph_transformer import GraphTransformer


device = torch.device('cuda')

def multiple_operator(a, b):
    return a * b

def rotate_operator(a, b):
    a_re, a_im = a.chunk(2, dim=-1)
    b_re, b_im = b.chunk(2, dim=-1)
    message_re = a_re * b_re - a_im * b_im
    message_im = a_re * b_im + a_im * b_re
    message = torch.cat([message_re, message_im], dim=-1)
    return message

class Multi_Head_Self_ATT(nn.Module):
    def __init__(self, head_num, in_channels, out_channels):
        super(Multi_Head_Self_ATT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_num = head_num

        self.k_lin = nn.Linear(self.in_channels, self.out_channels)
        self.q_lin = nn.Linear(self.in_channels, self.out_channels)
        self.v_lin = nn.Linear(self.in_channels, self.out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_lin.weight)
        nn.init.xavier_uniform_(self.q_lin.weight)
        nn.init.xavier_uniform_(self.v_lin.weight)
        nn.init.constant_(self.k_lin.bias, 0.0)
        nn.init.constant_(self.q_lin.bias, 0.0)
        nn.init.constant_(self.v_lin.bias, 0.0)

    def forward(self, x):
        B, M, _ = x.shape
        H, D = self.head_num, self.out_channels // self.head_num

        q = self.q_lin(torch.mean(x, dim = 1)).view(B, 1, H, D)
        k = self.k_lin(x).view(x.shape[:-1] + (H, D))
        v = x.view(x.shape[:-1] + (H, D))

        q = q.permute(0,2,1,3)
        k = k.permute(0,2,3,1)
        v = v.permute(0,2,1,3)

        alpha = F.softmax((q @ k / math.sqrt(q.size(-1))), dim=-1)

        o = alpha @ v

        output = o.permute(0,2,1,3).reshape((B, H * D)) 

        return output

class MyModel(nn.Module):
    def __init__(self, meta_g, drug_number, disease_number):
        super(MyModel, self).__init__()
        self.drug_number = drug_number
        self.meta_g = meta_g
        self.disease_number = disease_number

        self.drug_linear = nn.Linear(300, args.hgt_out_dim)
        self.disease_linear = nn.Linear(64, args.hgt_out_dim)

        self.gt_drug = GraphTransformer(device, args.gt_layer, self.drug_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_disease = GraphTransformer(device, args.gt_layer, self.disease_number, args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_out_dim, int(args.hgt_out_dim/args.hgt_head), args.hgt_head, len(self.meta_g.nodes()), len(self.meta_g.edges()), args.dropout)

        self.hgt = nn.ModuleList()
        for _ in range(args.hgt_layer):
            self.hgt.append(self.hgt_dgl)
        
        self.drug_self_att = Multi_Head_Self_ATT(args.gt_head, args.gt_out_dim, args.gt_out_dim)
        self.disease_self_att = Multi_Head_Self_ATT(args.gt_head, args.gt_out_dim, args.gt_out_dim)

        self.mlp = nn.Sequential(
        nn.Linear(args.gt_out_dim * 4, 1024),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(args.dropout),
        nn.Linear(256, 2))

    def forward(self, drdr_similarity_graph, didi_similarity_graph, drdr_dissimilarity_graph, didi_dissimilarity_graph, positive_heterograph, negative_heterograph, drug_feature, disease_feature, sample): 
        dr_sim_positive = self.gt_drug(drdr_similarity_graph)
        dr_sim_negative = self.gt_drug(drdr_dissimilarity_graph)
        
        di_sim_positive = self.gt_disease(didi_similarity_graph)
        di_sim_negative = self.gt_disease(didi_dissimilarity_graph)
        
        drug_feature = self.drug_linear(drug_feature)
        disease_feature = self.disease_linear(disease_feature)

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature
        }

        positive_heterograph.ndata['h'] = feature_dict
        positive_g = dgl.to_homogeneous(positive_heterograph, ndata='h')

        negative_heterograph.ndata['h'] = feature_dict
        negative_g = dgl.to_homogeneous(negative_heterograph, ndata='h')

        feature = torch.cat((drug_feature, disease_feature), dim=0)

        for layer in self.hgt:
            hgt_out = layer(positive_g, feature, positive_g.ndata['_TYPE'], positive_g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt_positive = hgt_out[:self.drug_number, :]
        di_hgt_positive = hgt_out[self.drug_number:, :]

        feature = torch.cat((drug_feature, disease_feature), dim=0)

        for layer in self.hgt:
            hgt_out = layer(negative_g, feature, negative_g.ndata['_TYPE'], negative_g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt_negative = hgt_out[:self.drug_number, :]
        di_hgt_negative = hgt_out[self.drug_number:, :]
        
        dr = torch.stack((dr_sim_positive, dr_sim_negative, dr_hgt_positive, dr_hgt_negative), dim=1)
        di = torch.stack((di_sim_positive, di_sim_negative, di_hgt_positive, di_hgt_negative), dim=1)

        dr_final = self.drug_self_att(dr)
        di_final = self.disease_self_att(di)

        dr_sample = dr_final[sample[:, 0]]
        di_sample = di_final[sample[:, 1]]

        m_result = multiple_operator(dr_sample, di_sample)
        r_result = rotate_operator(dr_sample, di_sample)
        drdi_embedding = torch.cat([dr_sample, di_sample, m_result, r_result], dim = 1)

        output = self.mlp(drdi_embedding)

        return output, (dr_final, dr_sim_positive, dr_sim_negative, dr_hgt_positive, dr_hgt_negative), (di_final, di_sim_positive, di_sim_negative, di_hgt_positive, di_hgt_negative)
    
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))
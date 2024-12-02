import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from model import MyModel
import torch.optim as optim
from parse_args import args
from metric import get_metric
import torch.nn.functional as fn
from torch_geometric.seed import seed_everything
from contrastive_learning import inter_contrastive
from data_preprocessing import process_data, dgl_heterograph

torch.autograd.set_detect_anomaly(True)

def train(model, drdr_similarity_graph, didi_similarity_graph, drdr_dissimilarity_graph, didi_dissimilarity_graph, positive_heterograph, negative_heterograph, drug_feature, disease_feature, X_train, X_test, Y_train, Y_test, drdr_matrix, didi_matrix):
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    best_metric = -float("inf")
    best_metrics = None
    l_reports = []

    for epoch in tqdm(range(args.total_epochs)):
        model.train()

        train_score, (dr_final, dr_sim_positive, dr_sim_negative, dr_hgt_positive, dr_hgt_negative), (di_final, di_sim_positive, di_sim_negative, di_hgt_positive, di_hgt_negative) = model(drdr_similarity_graph, didi_similarity_graph, drdr_dissimilarity_graph, didi_dissimilarity_graph, positive_heterograph, negative_heterograph, drug_feature, disease_feature, X_train)

        train_loss = cross_entropy(train_score, torch.flatten(Y_train))

        inter_CL_loss_positive = inter_contrastive(drdr_matrix, didi_matrix, dr_sim_positive, di_sim_positive, dr_hgt_positive, di_hgt_positive)

        inter_CL_loss_negative = inter_contrastive(drdr_matrix, didi_matrix, dr_sim_negative, di_sim_negative, dr_hgt_negative, di_hgt_negative)

        inter_CL_loss = inter_CL_loss_positive + inter_CL_loss_negative

        train_loss = train_loss + args.inter_ssl_reg * inter_CL_loss

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if use_cuda:
            torch.cuda.empty_cache()

        with torch.no_grad():
            model.eval()
            test_score, _, _ = model(drdr_similarity_graph, didi_similarity_graph, drdr_dissimilarity_graph, didi_dissimilarity_graph, positive_heterograph, negative_heterograph, drug_feature, disease_feature, X_test)
        
        test_prob = fn.softmax(test_score, dim=-1)
        test_score = torch.argmax(test_score, dim=-1)

        test_prob = test_prob[:, 1]
        test_prob = test_prob.cpu().numpy()

        test_score = test_score.cpu().numpy()

        AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

        current_result = [epoch + 1, round(AUC, 5), round(AUPR, 5), round(accuracy, 5), round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
        l_reports.append("\t".join(map(str, current_result)) + "\n")

        if epoch % 20 ==0:
            print("\t".join(map(str, current_result)))

        mix_factor = AUC + AUPR + mcc
        if mix_factor > best_metric:
            best_metric = mix_factor
            best_metrics = current_result
        
    return best_metrics

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    seed_everything(args.seed)

    if args.dataset in ["C-dataset", "F-dataset"]:
        args.lr = 2e-4
    
    print("dataset = {}, seed = {}, lr = {}".format(args.dataset, args.seed, args.lr))

    d_data, drdr_similarity_graph, didi_similarity_graph, drdr_dissimilarity_graph, didi_dissimilarity_graph = process_data()
    drdr_similarity_graph = drdr_similarity_graph.to(device)
    didi_similarity_graph = didi_similarity_graph.to(device)
    drdr_dissimilarity_graph = drdr_dissimilarity_graph.to(device)
    didi_dissimilarity_graph = didi_dissimilarity_graph.to(device)

    l_results = []

    for fold_i in range(args.K_fold):
        print('fold:', fold_i)

        positive_num = int(np.sum(d_data['Y_train'][fold_i] == 1))
        negative_num = d_data['Y_train'][fold_i].shape[0] - positive_num

        np_X_train_fold_i_positive = d_data['X_train'][fold_i][:int(args.dataset_percent * positive_num)]
        np_X_train_fold_i_negative = d_data['X_train'][fold_i][positive_num: positive_num + int(args.dataset_percent * negative_num)]
        np_X_train_fold_i = np.concatenate((np_X_train_fold_i_positive, np_X_train_fold_i_negative), axis=0)

        np_Y_train_fold_i_positive = d_data['Y_train'][fold_i][:int(args.dataset_percent * positive_num)]
        np_Y_train_fold_i_negative = d_data['Y_train'][fold_i][positive_num: positive_num + int(args.dataset_percent * negative_num)]
        np_Y_train_fold_i = np.concatenate((np_Y_train_fold_i_positive, np_Y_train_fold_i_negative), axis=0)


        X_train = torch.LongTensor(np_X_train_fold_i).to(device)
        Y_train = torch.LongTensor(np_Y_train_fold_i).to(device)
        X_test = torch.LongTensor(d_data['X_test'][fold_i]).to(device)
        Y_test = d_data['Y_test'][fold_i].flatten()

        positive_heterograph = dgl_heterograph(d_data, np_X_train_fold_i_positive)

        negative_heterograph = dgl_heterograph(d_data, np_X_train_fold_i_negative)

        positive_heterograph = positive_heterograph.to(device)
        meta_g = positive_heterograph.metagraph()
        negative_heterograph = negative_heterograph.to(device)

        model = MyModel(meta_g, d_data['drug_number'], d_data['disease_number'])
        model = model.to(device)

        drug_feature = torch.FloatTensor(d_data['drugfeature']).to(device)
        disease_feature = torch.FloatTensor(d_data['diseasefeature']).to(device)

        drdr_matrix_bool = d_data["drdr_matrix"]
        didi_matrix_bool = d_data["didi_matrix"]

        best_metrics = train(model, drdr_similarity_graph, didi_similarity_graph, drdr_dissimilarity_graph, didi_dissimilarity_graph, positive_heterograph, negative_heterograph, drug_feature, disease_feature, X_train, X_test, Y_train, Y_test, drdr_matrix_bool, didi_matrix_bool)

        if best_metrics is not None:
            l_results.append(best_metrics[1:])
            print(best_metrics)
    

    np_results = np.array(l_results)
    mean_result = np.round(np.mean(np_results, axis = 0), 4)

    print(mean_result)
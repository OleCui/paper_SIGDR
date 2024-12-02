import os
import sys
import argparse

argparser = argparse.ArgumentParser(sys.argv[0])
argparser.add_argument("--dataset",
                        type=str,
                        default = 'B-dataset',
                        help="dataset for training")

argparser.add_argument('--gpu', type=int, default=0,
                    help='gpu device')

argparser.add_argument('--seed', type=int, default=1234, help='random seed')

argparser.add_argument('--K_fold', type=int, default=10, help='k-fold cross validation')

argparser.add_argument('--negative_rate', type=float, default=1.0,help='negative_rate')

argparser.add_argument('--KNN_neighbor', type=int, default=20, help='neighbor_num')

argparser.add_argument('--total_epochs', type=int, default=1000,
                    help='adversarial learning epoch number.')

argparser.add_argument('--dropout', default=0.4, type=float, help='dropout')

argparser.add_argument('--gt_layer', default=2, type=int, help='graph transformer layer')

argparser.add_argument('--gt_head', default=4, type=int, help='graph transformer head')

argparser.add_argument('--gt_out_dim', default=256, type=int, help='graph transformer output dimension')

argparser.add_argument('--hgt_layer', default=2, type=int, help='heterogeneous graph transformer layer')

argparser.add_argument('--hgt_head', default=4, type=int, help='heterogeneous graph transformer head')

argparser.add_argument('--hgt_out_dim', default=256, type=int, help='heterogeneous graph transformer output dimension')

argparser.add_argument('--tr_layer', default=2, type=int, help='transformer layer')

argparser.add_argument('--tr_head', default=4, type=int, help='transformer head')

argparser.add_argument('--inter_ssl_temperature', type=float, default=0.05)

argparser.add_argument('--inter_ssl_reg', type=float, default=0.01)

argparser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')

argparser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

argparser.add_argument('--dataset_percent', type=float, default=1)

args = argparser.parse_args()
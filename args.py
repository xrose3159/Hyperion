import argparse
import os

supported_datasets = ['cora', 'citeseer', 'pubmed', 'physics', 'amazon_ratings']
supported_specified_task = ['cora']
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="citeseer")
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
parser.add_argument("--dataset_dir", type=str, default=root_dir)

log_path = os.path.join(os.path.dirname(current_path), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)
parser.add_argument("--logs_dir", type=str, default=log_path)
parser.add_argument("--specified_domain_skew_task", type=str, default=None)
parser.add_argument("--task", type=str, default="node_classification")
parser.add_argument("--skew_type", type=str, default="domain_skew")
parser.add_argument("--train_val_test_split", type=list, default=[0.6, 0.2, 0.2])
parser.add_argument("--dataset_split_metric", type=str, default="transductive")
parser.add_argument("--num_rounds", type=int, default=100)
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--cl_sample_rate", type=float, default=1.0)
parser.add_argument("--evaluation_mode", type=str, default="global")
parser.add_argument("--fed_algorithm", type=str, default="FedGNNProto")
parser.add_argument("--model", type=str, default="GNNPrototypeNet")
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.3)

parser.add_argument("--learning_rate", type=float, default=0.008)
parser.add_argument("--weight_decay", type=float, default=0.0002)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device_id", type=int, default=0)

parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
parser.add_argument("--least_samples", type=int, default=5)
parser.add_argument("--dirichlet_try_cnt", type=int, default=1000)

parser.add_argument('--noise_type', type=str, default='uniform', choices=['clean', 'pair', 'uniform'], help='Noise type')
parser.add_argument('--noise_rate', type=float, default=0.2, help='Noise rate')
parser.add_argument('--target_client_idx', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9], help='Target client index')

parser.add_argument('--mlp_out_dim', type=int, default=0,help='MLP output dim')
parser.add_argument('--batch_size', type=int, default=24,help='batch size')

parser.add_argument('--retain_ratio', type=float, default=0.95,help='retain ratio')
parser.add_argument('--pruning_epochs', type=int, default=50,help='pruning epochs')
#loss weights
parser.add_argument('--clst', type=float, default=0.0003,help='cluster')
parser.add_argument('--sep', type=float, default=0.9,help='separation')
parser.add_argument('--proto_contrast_weight', type=float, default=0.7,help='prototype contrastive weight')
parser.add_argument('--alpha', type=float, default=0.85,help='nodeselection weight')

parser.add_argument('--tau', type=float, default=0.7,help='prototype contrastive parameter')
parser.add_argument('--p', type=float, default=0.6,help='nodeselection paramater')

parser.add_argument('--prototype_model',  type=str, default="gcn")

parser.add_argument('--wkd_temperature', type=float, default=4.)
parser.add_argument('--sinkhorn_lambda', type=float, default=0.04)
parser.add_argument('--wkd_sinkhorn_iter', type=int, default=10)
parser.add_argument('--wkd_loss_cosine_decay_epoch', type=int, default=0)
parser.add_argument('--wkd_logit_weight', type=float, default=1.0)
parser.add_argument('--ce_loss_weight', type=float, default=1.0)
parser.add_argument('--solver_epochs', type=int, default=240)

parser.add_argument('--lr_decay_steps', type=int, default=30,help='decay_steps')
parser.add_argument('--lr_decay', type=float, default=0.5,help='decay rate')
parser.add_argument('--lr_min', type=float, default=1e-5,help='min learning rate')
parser.add_argument('--imb', type=int, default=0, help='imb')

args = parser.parse_args()

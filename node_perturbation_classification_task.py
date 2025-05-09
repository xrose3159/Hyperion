from base_task import BaseTask
from datasets.dataset_loader import load_dataset
from backbone import get_model, get_model_prototype
from algorithm import get_server, get_client
from datasets.partition import *
from datasets.processing import *
import numpy as np
from utils.logger import DefaultLogger
import time
from datasets.load_processed_data import *
from utils.Configures import model_args
from typing import Optional
import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes

class NodePerturbationClassificationTask(BaseTask):
    def __init__(self, args):
        super(NodePerturbationClassificationTask, self).__init__(args)
        self.server_data = None
        self.clients_data = None
        self.server = None
        self.clients = []
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.dataset = load_dataset(args.train_val_test_split, args.dataset_dir, args.dataset)
        self.input_dim = self.dataset.num_node_features
        listy = self.dataset.y.tolist()
        out_dim = len(np.unique(listy))
        self.num_classes = out_dim

    def process_data(self):
        set_random_seed(self.args.seed)
        self.dataset.n_classes = self.dataset.y.max() + 1 
        self.dataset.original_y = self.dataset.y.clone()
        clients_data, server_data = load_processed_data(self.args, self.dataset)
        target_client_idx = self.args.target_client_idx
        for idx in target_client_idx:
            target_client = clients_data[idx]  
            target_client.original_y = target_client.y.clone()
            train_labels = target_client.y[target_client.train_mask]
            noisy_train_labels, modified_mask = label_process(
                labels=train_labels, 
                n_classes=self.dataset.n_classes,
                noise_type=self.args.noise_type, 
                noise_rate=self.args.noise_rate,
                random_seed=self.args.seed,
                debug=True
            )
            target_client.y[target_client.train_mask] = noisy_train_labels
        for client in clients_data:
            if clients_data.index(client) not in target_client_idx: 
                client.original_y = client.y.clone()
        self.server_data = server_data
        self.clients_data = clients_data
    
    def init_server_client(self):
        clients = []
        for cid in range(self.args.num_clients):
            if(self.args.model == "GNNPrototypeNet"):
                client_model = get_model_prototype(self.args.model, self.input_dim, self.num_classes, model_args)
            else:
                client_model = get_model(self.args.model, self.input_dim, self.args.hidden_dim, self.num_classes,
                                     self.args.num_layers, self.args.dropout)
            client_model = client_model.to(self.device)
            client_data = self.clients_data[cid]
            client_data = client_data.to(self.device)
            client = get_client(self.args.fed_algorithm, self.args, client_model, client_data)
            clients.append(client)
        logger = DefaultLogger(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '-' + self.args.task + '-' +
                               self.args.dataset + '-' + self.args.fed_algorithm, self.args.logs_dir)
        if(self.args.model == "GNNPrototypeNet"):
            server_model = get_model_prototype(self.args.model, self.input_dim, self.num_classes, model_args)
        else:
            server_model = get_model(self.args.model, self.input_dim, self.args.hidden_dim, self.num_classes, self.args.num_layers, self.args.dropout)
        server_model = server_model.to(self.device)
        self.server_data = self.server_data.to(self.device)
        self.server = get_server(self.args.fed_algorithm, self.args, clients, server_model, self.server_data, logger)
        self.clients = clients
    
    def update_args(self):
        model_args.model_name = self.args.prototype_model
        model_args.mlp_out_dim = self.args.mlp_out_dim

    def run(self):
        self.process_data()
        self.update_args()
        self.init_server_client()
        test_acc, test_f1 = self.server.run()  
        return test_acc, test_f1  

def degree(index: Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> Tensor:
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)

def get_degree_array_for_pnaconv(dataset):
    deg_dict = {}
    for data in dataset:
        d = degree(data.edge_index[0], dtype=torch.long)
        for deg in d.tolist():
            if deg in deg_dict:
                deg_dict[deg] += 1
            else:
                deg_dict[deg] = 1
    degrees = sorted(deg_dict.items(), key=lambda x: x[1], reverse=True)
    n = 10  
    top_degrees = degrees[:n]
    degree_array = torch.tensor([deg for deg, count in top_degrees], dtype=torch.float32)
    return degree_array
from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn.functional as F
from utils.Configures import model_args
from sklearn.metrics import f1_score
from sklearn.mixture import GaussianMixture
import numpy as np
import math
import copy

class FedGNNProtoServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedGNNProtoServer, self).__init__(args, clients, model, data, logger)
        self.benign_clients = []
        self.poisoned_clients = []
        listy = self.data.y.tolist()
        self.num_classes = len(np.unique(listy))
        self.num_prototypes_per_class = model_args.num_prototypes_per_class
        self.prot_dim = model_args.prot_dim
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

    def detect_malicious_clients(self, clients_prototypes):
        num_clients = len(clients_prototypes)
        verbose = False  
        malicious_votes = np.zeros(num_clients)
        for vote_idx in range(9):  
            benign_scores = np.zeros(num_clients)
            for class_idx in range(self.num_classes):
                class_protos = []
                for client_idx, client_protos in enumerate(clients_prototypes):
                    proto = client_protos[class_idx].detach().cpu().numpy()
                    if proto.ndim > 2:
                        proto = proto.reshape(proto.shape[0], -1)
                    elif proto.ndim == 1:
                        proto = proto.reshape(1, -1)
                    proto_mean = np.mean(proto, axis=0)
                    class_protos.append(proto_mean)
                class_protos = np.array(class_protos)
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                class_protos_normalized = scaler.fit_transform(class_protos)
                if len(class_protos) <= 2 or np.all(np.std(class_protos_normalized, axis=0) < 1e-6):
                    continue
                gmm = GaussianMixture(
                    n_components=min(2, len(class_protos)-1), 
                    random_state=vote_idx,  
                    covariance_type='full',
                    max_iter=100,
                    n_init=3  
                )
                cluster_labels = gmm.fit_predict(class_protos_normalized)
                proba = gmm.predict_proba(class_protos_normalized)
                counts = np.bincount(cluster_labels)
                benign_cluster = np.argmax(counts)
                benign_scores += proba[:, benign_cluster]
            if np.all(benign_scores == 0):
                continue
                
            threshold = np.mean(benign_scores) - np.std(benign_scores)
            max_poisoned = int(num_clients * 0.4)
            sorted_indices = np.argsort(benign_scores)
            current_poisoned = sorted_indices[:min(max_poisoned, len(np.where(benign_scores < threshold)[0]))]
            malicious_votes[current_poisoned] += 1
        vote_threshold = 9 / 2  
        final_poisoned = np.where(malicious_votes > vote_threshold)[0].tolist()
        if not final_poisoned:
            max_select = int(num_clients * 0.3)
            sorted_by_votes = np.argsort(-malicious_votes)  
            final_poisoned = sorted_by_votes[:max_select].tolist()
            if np.all(malicious_votes[final_poisoned] == 0):
                final_poisoned = []
        self.poisoned_clients = final_poisoned
        self.benign_clients = list(set(range(num_clients)) - set(final_poisoned))
        return self.benign_clients, self.poisoned_clients

    def aggregate_prototype(self, clients_prototypes, benign_clients):
        aggregated_prototypes = {}
        for class_idx in range(self.num_classes):
            class_protos_list = []
            for client_idx in benign_clients:
                class_protos_list.append(clients_prototypes[client_idx][class_idx])
            class_protos = torch.cat(class_protos_list, dim=0)
            gmm = GaussianMixture(
                n_components=self.num_prototypes_per_class,
                random_state=42,
                covariance_type='full'
            )
            gmm.fit(class_protos.numpy())
            aggregated_prototypes[class_idx] = torch.tensor(gmm.means_).clone().detach().requires_grad_(True)
        return aggregated_prototypes

    def aggregate(self, benign_clients):
        num_total_samples = sum([self.clients[cid].num_samples for cid in benign_clients])
        for i, cid in enumerate(benign_clients):
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param

    def distill_server_by_prototype(self, initial_global_prototypes, benign_prototypes, poisoned_prototypes, cost_matrix, lr=0.01, max_steps=3, **kwargs):
        self.temperature = self.args.wkd_temperature   
        self.sinkhorn_lambda = self.args.sinkhorn_lambda
        self.sinkhorn_iter = self.args.wkd_sinkhorn_iter 
        self.loss_cosine_decay_epoch = self.args.wkd_loss_cosine_decay_epoch 
        self.wkd_logit_loss_weight = self.args.wkd_logit_weight 
        self.solver_epochs = self.args.solver_epochs 
        decay_start_epoch = self.loss_cosine_decay_epoch
        if kwargs['epoch'] > decay_start_epoch:
            # cosine decay
            self.wkd_logit_loss_weight_1 = 0.5*self.wkd_logit_loss_weight*(1+math.cos((kwargs['epoch']-decay_start_epoch)/(self.solver_epochs-decay_start_epoch)*math.pi))
        else:
            self.wkd_logit_loss_weight_1 = self.wkd_logit_loss_weight
        trainable_prototypes = {}
        for class_idx in range(self.num_classes):
            trainable_prototypes[class_idx] = initial_global_prototypes[class_idx].clone().detach().requires_grad_(True)
        logits_teacher = torch.stack([trainable_prototypes[class_idx] for class_idx in range(self.num_classes)]).reshape(self.num_classes * self.num_prototypes_per_class, self.prot_dim).to(torch.float32)
        benign_prototypes_reshaped = self.convert_prototypes_to_tensor(benign_prototypes)
        poisoned_prototypes_reshaped = self.convert_prototypes_to_tensor(poisoned_prototypes)
        optimizer = torch.optim.Adam([proto for proto in trainable_prototypes.values()], lr)
        for step in range(max_steps):  
            total_loss = 0.0
            for class_idx in range(self.num_classes * self.num_prototypes_per_class):
                optimizer.zero_grad()
                for logits_good_student in benign_prototypes_reshaped:
                    loss_wkd_logit = wkd_prototype_loss(
                        logits_good_student[class_idx], 
                        logits_teacher[class_idx], 
                        self.temperature,
                        self.wkd_logit_loss_weight_1, 
                        cost_matrix, 
                        self.sinkhorn_lambda, 
                        self.sinkhorn_iter
                    )
                    total_loss += loss_wkd_logit
                for logits_bad_student in poisoned_prototypes_reshaped:
                    loss_wkd_logit = wkd_prototype_loss(
                        logits_bad_student[class_idx], 
                        logits_teacher[class_idx], 
                        self.temperature,
                        self.wkd_logit_loss_weight_1, 
                        cost_matrix, 
                        self.sinkhorn_lambda, 
                        self.sinkhorn_iter
                    )
                    total_loss -= loss_wkd_logit  
                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_([proto for proto in trainable_prototypes.values()], max_norm=1.0)
                optimizer.step()
                total_loss = 0.0
        optimized_prototypes = {}
        for class_idx in range(self.num_classes):
            proto = trainable_prototypes[class_idx].detach()
            proto_norm = torch.norm(proto, p=2, dim=1, keepdim=True) 
            optimized_prototypes[class_idx] = (proto / (proto_norm + 1e-8)).float()
        return optimized_prototypes

    def convert_prototypes_to_tensor(self, prototypes_list):
        converted_prototypes = []
        for client_protos in prototypes_list:
            proto_tensors = []
            for class_idx in range(self.num_classes):
                proto_tensors.append(client_protos[class_idx])
            client_proto_tensor = torch.cat(proto_tensors, dim=0) 
            client_proto_tensor = client_proto_tensor.reshape(
                self.num_classes * self.num_prototypes_per_class, 
                client_proto_tensor.shape[-1]
            ).to(torch.float32)
            converted_prototypes.append(client_proto_tensor)
        return converted_prototypes

    def update_global_model(self, global_prototypes):
        prototype_vectors = []
        for class_idx in range(self.num_classes):
            prototype_vectors.append(global_prototypes[class_idx])
        prototype_vectors = torch.cat(prototype_vectors, dim=0).to(torch.float32)
        self.model.prototype_layer.prototype_vectors.data = prototype_vectors
    
    def communicate(self):
        for client in self.clients:
            for server_param, client_param in zip(self.model.parameters(), client.model.parameters()):
                client_param.data.copy_(server_param.data)

    def run(self):
        for round in range(self.num_rounds):
            print(f"Round {round+1}:")
            self.logger.write_round(round+1)
            self.communicate()
            avg_train_loss = 0
            clients_prototypes = []
            print("Training clients:\n", end=' ')
            for cid in range(len(self.clients)):
                print(cid, end=' ')
                client_loss = 0
                for epoch in range(self.num_epochs):
                    self.clients[cid].round = round
                    train_stats = self.clients[cid].train(epoch, round, self.num_classes)
                    client_loss += train_stats['total_loss']
                avg_train_loss += client_loss * self.clients[cid].num_samples / self.num_total_samples
                clients_prototypes.append(self.clients[cid].get_prototypes())  

                print(f"client {cid} loss: {client_loss:.4f}")
            print(f"\nAverage train loss: {avg_train_loss:.4f}")

            benign_clients, poisoned_clients = self.detect_malicious_clients(clients_prototypes)
            benign_prototypes = [clients_prototypes[i] for i in benign_clients]
            poisoned_prototypes = [clients_prototypes[i] for i in poisoned_clients]
            if(round > self.args.pruning_epochs and round % 5 == 0):
                for cid in poisoned_clients:
                    client = self.clients[cid]
                    with torch.no_grad(): 
                        pred, virtual_label, prot_nce_loss, graph_emb, distances = client.model(client.data)  
                    prune_training_set(client.data, client.model, distances, self.args.retain_ratio, model_args, self.device)      
            initial_global_prototypes = self.aggregate_prototype(clients_prototypes, benign_clients)
            self.aggregate(benign_clients)
            for client_idx in range(len(clients_prototypes)):
                for class_idx in clients_prototypes[client_idx]:
                    clients_prototypes[client_idx][class_idx] = clients_prototypes[client_idx][class_idx].to(self.device)
            for class_idx in initial_global_prototypes:            
                initial_global_prototypes[class_idx] = initial_global_prototypes[class_idx].to(self.device)
            cost_matrix = calc_cost_matrix_proto(initial_global_prototypes, self.num_prototypes_per_class, self.device)
            optimized_global_prototypes = self.distill_server_by_prototype(initial_global_prototypes, benign_prototypes, poisoned_prototypes, cost_matrix, lr=0.005, epoch=round)
            self.update_global_model(optimized_global_prototypes)
            for client in self.clients:
                client.update_prototypes(optimized_global_prototypes)
            self.local_validate()
            self.local_evaluate()
            self.global_evaluate()
        self.model.eval()
        with torch.no_grad():
            pred, virtual_label, prot_nce_loss, graph_emb, distances = self.model(self.data)
            test_pred = pred[self.data.test_mask].argmax(dim=1)
            test_true = self.data.y[self.data.test_mask]
            test_acc = test_pred.eq(test_true).float().mean().item()
            test_f1 = f1_score(test_true.cpu().numpy(), 
                            test_pred.cpu().numpy(), 
                            average='macro')
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        print(f"Final Test F1 Score: {test_f1:.4f}")
        return test_acc, test_f1 

class FedGNNProtoClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedGNNProtoClient, self).__init__(args, model, data)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.clst = args.clst
        self.sep = args.sep 
        self.proto_contrast_weight = args.proto_contrast_weight
        self.alpha = args.alpha
        self.num_classes = data.y.max().item() + 1
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.init_lr = args.learning_rate
        self.current_lr = self.init_lr
        self.lr_decay = args.lr_decay 
        self.lr_decay_steps = args.lr_decay_steps 
        self.lr_min = args.lr_min 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.current_lr, weight_decay=args.weight_decay)
        listy = self.data.y.tolist()
        self.num_classes = len(np.unique(listy))
        self.num_prototypes_per_class = model_args.num_prototypes_per_class
        self.tau = args.tau 
        self.p = args.p 
        self.imb = args.imb

    def update_learning_rate(self, round_number):
        if round_number % self.lr_decay_steps == 0 and round_number > 0:
            self.current_lr = max(self.init_lr * (self.lr_decay ** (round_number // self.lr_decay_steps)), self.lr_min)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
    def train(self, epoch, round, num_classes):
        self.update_learning_rate(round)
        self.model.train()
        self.num_classes = num_classes
        self.optimizer.zero_grad()
        self.epoch = epoch
        data = self.data.to(self.device)
        data1 = copy.deepcopy(data)
        data2 = copy.deepcopy(data)
        data1.edge_index, _ = dropout_adj(data.edge_index, p=self.p)
        data1.x = mask_feature(data.x, p=self.p)[0]
        data2.edge_index, _ = dropout_adj(data.edge_index, p=self.p)
        data2.x = mask_feature(data.x, p=self.p)[0]        
        pred, virtual_label, prot_nce_loss, graph_emb, distances = self.model(data)
        pred1, virtual_label1, prot_nce_loss1, graph_emb1, distances1 = self.model(data1)
        prototypes_of_correct_class = torch.t(self.model.get_prototype_class_identity().to(self.device)[:, data.y[data.train_mask].to(self.device)].bool()).to(self.device)
        train_distances = distances[data.train_mask]
        cluster_cost = torch.mean(torch.min(train_distances[prototypes_of_correct_class].reshape(-1, self.num_prototypes_per_class), dim=1)[0])
        eps = 1e-6
        separation_cost = torch.mean(1.0 / (torch.min(train_distances[~prototypes_of_correct_class].reshape(-1, (self.num_classes - 1) * self.num_prototypes_per_class), dim=1)[0] + eps))
        prototype_vectors = self.model.get_prototype_vectors()
        prototype_contrastive_loss = self.prototype_contrastive_loss(prototype_vectors)
        dynamic_ce_loss = self.dynamic_cross_entropy_loss(
            pred1[data.train_mask], 
            pred[data.train_mask], 
            data.y[data.train_mask]
        )
        total_loss = self.clst * cluster_cost + self.sep * separation_cost + \
                    self.proto_contrast_weight * prototype_contrastive_loss + self.alpha * dynamic_ce_loss
        total_loss.backward()
        self.optimizer.step()
        _, prediction = torch.max(pred, -1)
        if self.imb is True:
            batch_f1_macro = f1_score(data.y.cpu().numpy(), prediction.cpu().numpy(), average='macro')
            acc = batch_f1_macro
        else:
            acc = prediction.eq(data.y).cpu().numpy().mean()
        return {
            'loss': total_loss.item(),
            'acc': acc,
            'total_loss': total_loss.item(),
            'graph_emb': graph_emb.detach().cpu(),
            'y': data.y.cpu(),
            'prototype_contrastive_loss': prototype_contrastive_loss if isinstance(prototype_contrastive_loss, float) else prototype_contrastive_loss.item(),
            'dynamic_ce_loss': dynamic_ce_loss.item()
        }

    def prototype_contrastive_loss(self, prototype_vectors):
        num_prototypes = prototype_vectors.shape[0]
        prototype_vectors_norm = F.normalize(prototype_vectors, p=2, dim=1)
        sim_matrix = torch.mm(prototype_vectors_norm, prototype_vectors_norm.t()) / self.tau
        same_class_mask = torch.zeros((num_prototypes, num_prototypes), device=prototype_vectors.device)
        for class_idx in range(self.num_classes):
            start_idx = class_idx * self.num_prototypes_per_class
            end_idx = (class_idx + 1) * self.num_prototypes_per_class
            same_class_mask[start_idx:end_idx, start_idx:end_idx] = 1
        same_class_mask = same_class_mask - torch.eye(num_prototypes, device=prototype_vectors.device)
        diff_class_mask = 1 - same_class_mask - torch.eye(num_prototypes, device=prototype_vectors.device)
        loss = 0
        for i in range(num_prototypes):
            pos_indices = torch.where(same_class_mask[i] > 0)[0]
            if len(pos_indices) == 0:
                continue
            pos_sim = torch.exp(sim_matrix[i, pos_indices])
            neg_indices = torch.where(diff_class_mask[i] > 0)[0]
            neg_sim = torch.sum(torch.exp(sim_matrix[i, neg_indices]))
            curr_loss = -torch.log(pos_sim.sum() / (pos_sim.sum() + neg_sim + 1e-8))
            loss += curr_loss
        loss = loss / num_prototypes
        return loss

    def dynamic_cross_entropy_loss(self, p1, p2, labels):
        labels = labels.long()
        pseudo_labels1 = p1.argmax(dim=1)
        pseudo_labels2 = p2.argmax(dim=1)
        consistent_mask = (pseudo_labels1 == pseudo_labels2)
        if not consistent_mask.any():
            return F.cross_entropy(p1, labels)
        loss = F.cross_entropy(p1[consistent_mask], labels[consistent_mask])
        return loss

    def get_prototypes(self):
        prototype_vectors = self.model.get_prototype_vectors()
        prototype_class_identity = self.model.get_prototype_class_identity()
        prototypes_by_class = {}
        num_classes = prototype_class_identity.size(1)
        num_prototypes_per_class = self.num_prototypes_per_class
        for class_idx in range(num_classes):
            start_idx = class_idx * num_prototypes_per_class
            end_idx = (class_idx + 1) * num_prototypes_per_class
            class_prototypes = prototype_vectors[start_idx:end_idx].detach().cpu()
            prototypes_by_class[class_idx] = class_prototypes
        return prototypes_by_class

    def update_prototypes(self, global_prototypes):
        prototype_vectors = []
        for class_idx in range(self.num_classes):
            prototype_vectors.append(global_prototypes[class_idx])
        prototype_vectors = torch.cat(prototype_vectors, dim=0)
        self.model.prototype_layer.prototype_vectors.data = prototype_vectors.to(self.device)

def sinkhorn(w1, w2, cost, reg=0.05, max_iter=10):
    bs, dim = w1.shape
    w1 = w1.unsqueeze(-1)
    w2 = w2.unsqueeze(-1)
    u = 1/dim*torch.ones_like(w1, device=w1.device, dtype=w1.dtype) 
    K = torch.exp(-cost / reg)
    Kt= K.transpose(2, 1)
    for i in range(max_iter):
        v=w2/(torch.bmm(Kt,u)+1e-8) 
        u=w1/(torch.bmm(K,v)+1e-8)  
    flow = u.reshape(bs, -1, 1) * K * v.reshape(bs, 1, -1)
    return flow

def calc_cost_matrix(initial_global_prototypes, num_prototypes_per_class, device):
    prototypes = initial_global_prototypes.to(device)
    num_classes = prototypes.shape[0] // num_prototypes_per_class
    feature_dim = prototypes.shape[1]
    prototypes_reshaped = prototypes.reshape(num_classes, num_prototypes_per_class * feature_dim)  
    proto_normed = F.normalize(prototypes_reshaped, p=2, dim=-1) 
    cosine_sim = proto_normed.matmul(proto_normed.transpose(-1, -2))  
    cost_matrix = 1 - cosine_sim
    return cost_matrix

def wkd_prototype_loss(logits_student, logits_teacher, temperature, gamma, cost_matrix=None, sinkhorn_lambda=0.05, sinkhorn_iter=10):
    if logits_student.dim() == 1:
        logits_student = logits_student.unsqueeze(0)  
    if logits_teacher.dim() == 1:
        logits_teacher = logits_teacher.unsqueeze(0)  
    logits_student = logits_student.to(torch.float32)
    logits_teacher = logits_teacher.to(torch.float32)
    pred_student = F.softmax(logits_student / temperature, dim=-1).to(torch.float32)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=-1).to(torch.float32)
    cost_matrix = F.relu(cost_matrix) + 1e-8
    if cost_matrix.dim() == 2:
        cost_matrix = cost_matrix.unsqueeze(0)
    cost_matrix = cost_matrix.to(torch.float32).to(pred_student.device)
    flow = sinkhorn(pred_student, pred_teacher, cost_matrix, reg=sinkhorn_lambda, max_iter=sinkhorn_iter)
    ws_distance = (flow * cost_matrix).sum(-1).sum(-1)
    ws_distance = ws_distance.mean()
    return gamma * ws_distance

def calc_cost_matrix_proto(initial_global_prototypes, num_prototypes_per_class, device):
    prototypes_tensor = torch.stack(
        [proto.to(device) for proto in initial_global_prototypes.values()])  
    prototypes_flat = prototypes_tensor.reshape(-1, prototypes_tensor.shape[-1])  
    def stable_cosine_distance(x):
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-8)  
        sim_matrix = torch.mm(x_norm, x_norm.T)       
        return 1 - sim_matrix                         
    cost_matrix = stable_cosine_distance(prototypes_flat.T)  
    cost_matrix.fill_diagonal_(0.0)  
    cost_matrix = torch.clamp(cost_matrix, min=0, max=2)  
    return cost_matrix

def prune_training_set(data, model, distances, retain_ratio, model_args, device):
    train_indices = torch.where(data.train_mask)[0]
    node_prototype_bool = get_node_prototype_bool_matrix(data, model, device)
    min_distances = torch.zeros(distances.shape[0], device=device)
    for i in range(distances.shape[0]):
        class_proto_dist = distances[i][node_prototype_bool[i]]
        if class_proto_dist.numel() > 0:
            min_distances[i] = torch.min(class_proto_dist)
        else:
            min_distances[i] = 1000.0
    train_min_distances = min_distances[train_indices]
    sorted_train_indices = torch.argsort(train_min_distances)  
    retain_count = max(1, int(len(train_min_distances) * retain_ratio))  
    top_train_indices = sorted_train_indices[:retain_count]
    retained_train_indices = train_indices[top_train_indices]
    new_train_mask = torch.zeros_like(data.train_mask, dtype=torch.bool)
    new_train_mask[retained_train_indices] = True
    data.train_mask = new_train_mask
    return data

def get_node_prototype_bool_matrix(data, model, device):
    node_labels = data.y.to(device)  
    prototype_class_identity = model.get_prototype_class_identity().to(device)  
    node_prototype_bool = prototype_class_identity[:, node_labels].bool().t() 
    return node_prototype_bool









































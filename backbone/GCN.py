import torch
import torch.nn.functional as F

from torch.nn import ModuleList

def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


class GCN(torch.nn.Module):
    def __init__(self, input_dim, model_args):
        super().__init__()
        self.hidden_channels = 384
        self.out_channels = model_args.prot_dim
        self.conv1 = GCNConv(input_dim, self.hidden_channels,
                             normalize=True)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels,
                             normalize=True)
        self.conv3 = GCNConv(self.hidden_channels, self.hidden_channels,
                             normalize=True)
        self.conv4 = GCNConv(self.hidden_channels, self.out_channels,
                             normalize=True)
        self.conv_layers = ModuleList()
        self.conv_layers.append(self.conv1)
        self.conv_layers.append(self.conv2)
        self.conv_layers.append(self.conv3)
        self.conv_layers.append(self.conv4)
        self.readout_layers = get_readout_layers(model_args.readout)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        edge_weight = None
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, edge_index, edge_weight)
        # pooled = [readout(x, batch) for readout in self.readout_layers]
        # x = torch.cat(pooled, dim=-1)
        return x
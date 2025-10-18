# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import math
from torch.nn.parameter import Parameter
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool, global_max_pool


###########################################################
#                       Edge attr                         #
###########################################################
# GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, use_edge_attr=True, num_edge_attr=7):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr = "add")
        self.use_edge_attr = use_edge_attr
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0.001]))
        if self.use_edge_attr:
            self.edge_encoder = torch.nn.Linear(num_edge_attr, emb_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if self.use_edge_attr:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        if self.use_edge_attr:
            return F.relu(x_j + edge_attr)
        else:
            return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, use_edge_attr=True, num_edge_attr=7):
        super(GCNConv, self).__init__(aggr='add')

        self.use_edge_attr = use_edge_attr
        self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                          torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if self.use_edge_attr:
            self.edge_encoder = torch.nn.Linear(num_edge_attr, emb_dim)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.linear(x)
        if self.use_edge_attr:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None
        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

        return out

    def message(self, x_j, edge_attr, norm):
        if self.use_edge_attr:
            return norm.view(-1, 1) * F.relu(x_j + edge_attr)
        else:
            return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


# convolution for injection
class PGCNConv(MessagePassing):
    def __init__(self, emb_dim, use_edge_attr=False, num_edge_attr=7):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(PGCNConv, self).__init__(aggr = "add")
        self.use_edge_attr = use_edge_attr
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0.001]))
        if self.use_edge_attr:
            self.edge_encoder = torch.nn.Linear(num_edge_attr, emb_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if self.use_edge_attr:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm))

        return out

    def message(self, x_j, edge_attr, norm):
        if self.use_edge_attr:
            return norm.view(-1, 1) * F.relu(x_j + edge_attr)
        else:
            return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


# PDM convolution
class PEMConv(MessagePassing):
    def __init__(self, emb_dim, use_edge_attr=False, num_edge_attr=7):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(PEMConv, self).__init__(aggr = "add")
        self.use_edge_attr = use_edge_attr
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim), torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0.001]))
        if self.use_edge_attr:
            self.edge_encoder = torch.nn.Linear(num_edge_attr, emb_dim)

    def forward(self, x, edge_index, edge_attr=None):
        if self.use_edge_attr:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = None

        col, row = edge_index
        N = x.shape[0]
        adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))  # .to_device(x.device)
        deg = adj.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        weights = adj.sum(dim=1).to(torch.float) + self.eps

        if self.use_edge_attr:
            out = self.mlp(
                weights.view(-1, 1) * x + adj @ x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        else:
            out = self.mlp(weights.view(-1, 1) * x + adj @ x)

        return out

    def message(self, x_j, edge_attr):
        if self.use_edge_attr:
            return F.relu(edge_attr)
        else:
            return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out


###########################################################
#                      Permutation                        #
###########################################################
# Permutation Invariance Model
class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.use_edge_attr = args.use_edge_attr
        self.num_edge_attr = args.num_edge_attr
        if self.use_edge_attr and self.num_edge_attr != 0:
            self.use_edge_attr = True
        else:
            self.use_edge_attr = False

        self.deg_emb = torch.nn.Embedding(100, self.emb_dim//2)
        self.node_encoder = torch.nn.Sequential(nn.Linear(self.num_features, self.emb_dim//2))
        self.mlp = torch.nn.Sequential(torch.nn.BatchNorm1d(self.emb_dim), torch.nn.ReLU())
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for i in range(self.num_layer):
            if args.gnn == 'gin':
                self.convs.append(GINConv(emb_dim=self.emb_dim, use_edge_attr=self.use_edge_attr, num_edge_attr=self.num_edge_attr))
            elif args.gnn == 'gcn':
                self.convs.append(GCNConv(emb_dim=self.emb_dim, use_edge_attr=self.use_edge_attr, num_edge_attr=self.num_edge_attr))
            elif args.gnn == 'pgcn':
                self.convs.append(PGCNConv(emb_dim=self.emb_dim, use_edge_attr=self.use_edge_attr, num_edge_attr=self.num_edge_attr))
            elif args.gnn == 'pem':
                self.convs.append(PEMConv(emb_dim=self.emb_dim, use_edge_attr=self.use_edge_attr, num_edge_attr=self.num_edge_attr))
            self.batch_norms.append(torch.nn.BatchNorm1d(args.emb_dim))
        self.net = Network(num_layer=1, emb_dim=self.emb_dim, num_classes=self.num_classes)

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, \
                                          batched_data.edge_attr, batched_data.batch

        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype).to(torch.long) % 100

        h_list = [self.mlp(torch.cat((self.node_encoder(x), self.deg_emb(deg)), dim=1))]
        node_representation = 0
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer != self.num_layer - 1:
                h = F.relu(h)

            h_list.append(h)
            node_representation += h

        node_representation, _ = to_dense_batch(node_representation, batch)
        x_img = (node_representation.transpose(1, 2) @ node_representation).unsqueeze(1)
        out = self.net(x_img)

        return out


###########################################################
#                      Mat Pooling                        #
###########################################################
class Network(torch.nn.Module):
    def __init__(self, num_layer=3, emb_dim=256, num_classes=10):
        super(Network, self).__init__()

        self.num_classes = num_classes
        self.mlp = nn.Sequential(
            MatNet(channels=num_layer, left_in=emb_dim, left_out=emb_dim//2, right_in=emb_dim, right_out=emb_dim//2),
            nn.BatchNorm2d(num_layer),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((emb_dim // 4, emb_dim // 4)),
            MatNet(channels=num_layer, left_in=emb_dim//4, left_out=emb_dim//16, right_in=emb_dim//4, right_out=emb_dim//16),
            nn.BatchNorm2d(num_layer),
        )
        self.classifier = nn.Linear(emb_dim//16 * emb_dim//16 * num_layer, self.num_classes)

    def forward(self, img):
        x = self.mlp(img)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x


class MatNet(torch.nn.Module):
    def __init__(self, channels: int, left_in: int, left_out: int, right_in: int, right_out: int, bias: bool = True, dot: bool = True) -> None:
        super(MatNet, self).__init__()

        self.channels = channels
        self.L_in = left_in
        self.L_out = left_out
        self.R_in = right_in
        self.R_out = right_out

        if dot:
            self.dot = Parameter(torch.Tensor(channels, self.L_in, self.R_in))
            nn.init.uniform_(self.dot, -1, 1)
        else:
            self.register_parameter('dot', None)

        self.left_w = Parameter(torch.Tensor(channels, self.L_out, self.L_in))
        self.right_w = Parameter(torch.Tensor(channels, self.R_in, self.R_out))
        if bias:
            self.bias = Parameter(torch.Tensor(channels, self.L_out, self.R_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.left_w, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.right_w, a=math.sqrt(5))
        if self.bias is not None:
            fan_left, _ = nn.init._calculate_fan_in_and_fan_out(self.left_w)
            fan_right, _ = nn.init._calculate_fan_in_and_fan_out(self.right_w)
            bound = 1 / math.sqrt((fan_left+fan_right)/2)
            # bound = 1 / math.sqrt(fan_left)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # out = self.left_w@input@self.right_w + self.bias
        if self.dot != None:
            input = input * self.dot
        out = self.left_w @ input @ self.right_w + self.bias

        return out
from cgi import test
import os
from turtle import forward, hideturtle
from sklearn import preprocessing
import torch 
import numpy as np
import argparse
import scipy.sparse as sp
import torch.nn.functional as F
from torch.nn import Parameter

from data_processor import Dataset
def sparse_dropout(x, drop_prob, noise_shape, training):
    if (not training) or (drop_prob == 0):
        return x
    noise = torch.empty(noise_shape)
    torch.nn.init.uniform_(noise)
    keep_mask = torch.le(noise, 1 - drop_prob)
    indices, values, shape_sparse = x._indices(), x._values(), x.size()
    indices = indices[:, keep_mask]
    values= values[keep_mask] * (1. / (1 - drop_prob))
    res = torch.sparse_coo_tensor(indices, values, shape_sparse)
    return res
    
def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)
    return res
class GraphConv(torch.nn.Module):
    def __init__(self, 
                    input_dim, output_dim,
                    num_features_nonzero, dropout: float =0.5,
                    act=F.relu, has_bias: bool = True):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features_nonzero = num_features_nonzero
        self.act = act
        self.has_bias = has_bias
        self.dropout_rate = dropout

        self.weight = Parameter(torch.empty((input_dim, output_dim)))
        if has_bias:
            self.bias = Parameter(torch.empty(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    def forward(self, x, adj):
        if x.is_sparse:
            x = sparse_dropout(x, drop_prob=self.dropout_rate, noise_shape=self.num_features_nonzero, training=self.training) 
        else:
            x = F.dropout(x, self.dropout_rate, training=self.training)
        x = dot(x, self.weight)
        x = dot(adj, x, sparse=True)
        if self.has_bias:
            x = x + self.bias
        x = self.act(x)
        return x
    
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
class GCN(torch.nn.Module):
    
    def __init__(self, 
                    input_dim, hidden_dim, output_dim, num_features_nonzero,
                    dropout_rate, act=F.relu, has_bias: bool=True,
                    weight_decayed: float=0.):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_features_nonzero = num_features_nonzero
        self.dropout = dropout_rate
        self.weight_decayed = weight_decayed

        self.layer1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim,
                            num_features_nonzero=self.num_features_nonzero,
                            dropout=dropout_rate,
                            act=act, has_bias=has_bias) 

        self.layer2 = GraphConv(
                            input_dim=hidden_dim, output_dim=output_dim,
                            num_features_nonzero=self.num_features_nonzero,
                            dropout=dropout_rate,
                            act=act, has_bias=has_bias) 
    
    def l2_reg_layer1(self):
        l2 = torch.tensor(0.) 
        for p in self.layer1.parameters():
            p = torch.reshape(p, shape=(-1,))
            l2  = l2 + torch.linalg.vector_norm(p)
        l2 = self.weight_decayed * l2
        return l2

    def forward(self, x, adj):
        hidden = self.layer1(x, adj)
        output = self.layer2(hidden, adj)
        return F.log_softmax(output, dim=-1)


def train_model(x, labels, adj, train_mask, model, opti): 
    model.train()
    pred = model(x, adj)
    loss = F.nll_loss(pred[train_mask], labels[train_mask]) + model.l2_reg_layer1()
    opti.zero_grad()
    loss.backward()
    opti.step()
    return loss

def eval_model(x, labels, adj,  test_mask, train_mask, model):
    model.eval()
    pred = model(x, adj)
    pred = torch.argmax(pred, dim=-1)
    test_acc =  torch.sum(torch.eq(pred[test_mask], labels[test_mask]).type(torch.float32)) / torch.sum(test_mask.type(torch.float32))
    train_acc =  torch.sum(torch.eq(pred[train_mask], labels[train_mask]).type(torch.float32)) / torch.sum(train_mask.type(torch.float32))
    return train_acc, test_acc
    

if __name__ == "__main__":
    # hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_str", default="cora", type=str)
    parser.add_argument("--hidden1", default=16, type=int)
    parser.add_argument("--num_epoch", default=600, type=int)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--weight_decayed", default=5e-4, type=float)
    #parser.add_argument("--weight_decayed", default=0., type=float)
    args = parser.parse_args()

    # dataset process class
    dataset = Dataset(args)
    output = dataset.load_data()
    adj, features, labels, train_mask, val_mask, test_mask = output

    features = dataset.preprocess_features(features)
    adj = dataset.preprocess_adj(adj)
    num_features_nonzero = len(features[1])
    num_classes = labels.shape[-1]
    input_features_dim = features[2][1]

    gcn = GCN(input_dim=input_features_dim, hidden_dim=args.hidden1, output_dim=num_classes, 
                        num_features_nonzero=num_features_nonzero,
                        dropout_rate=args.dropout, weight_decayed=args.weight_decayed)
    labels = torch.tensor(labels)
    labels = torch.argmax(labels, dim=-1)
    print("============")
    features = torch.sparse_coo_tensor(features[0], features[1], features[2])
    adj = torch.sparse_coo_tensor(adj[0], adj[1], adj[2]).type(features.dtype)

    train_mask = torch.tensor(train_mask)
    test_mask = torch.tensor(test_mask)
    val_mask = torch.tensor(val_mask)


    opti = torch.optim.Adam(gcn.parameters(), lr=0.01)

    f = open("metrics_05.csv", "wt")
    print("loss\ttrain_acc\ttest_acc", file=f)
    for epoch in range(args.num_epoch):
        loss = train_model(features, labels, adj, train_mask, gcn, opti)
        train_acc, test_acc = eval_model(features, labels, adj, test_mask, train_mask, gcn)
        print(f"loss: {loss:>7f}, train_acc: {train_acc:>7f}, test_acc: {test_acc:>7f}, [{epoch:>5d}/{args.num_epoch:>5d}]")
        save_str = f"{loss:7f}\t{train_acc:7f}\t{test_acc:7f}"
        print(save_str, file=f)
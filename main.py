"""
Thanks Yucheng Shi, Kaixiong Zhou, Ninghao Liu public their work 
'ENGAGE: Explanation Guided Data Augmentation for Graph Representation Learning'

Parts of implementation are reference to 'ENGAGE'
"""
import torch
import numpy as np
import dataloader
import AUL
import utils
from torch_geometric.loader import DataLoader
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from tqdm import tqdm
import argparse
from evaluation import mlp_evaluator
from braGCL import EncoderGCN, Model
from data_aug import dropout_edge_guided, drop_feature_guided


def train(model, optimizer, dataloader, estimator, epoch):
    model.train()
    optimizer.zero_grad()
    for data in dataloader:
        data = data.to(device)
        if epoch < int(args.start_e*num_epochs):
            vote = None
            nodevote = None
        else:
            vote, nodevote = get_expl(model, data)
        edge_index_1, edge_index_2 = dropout_edge_guided(data.edge_index, vote, edge_p = edge_p, lambda_edge= lambda_edge)

        if data.x is None:
            if epoch > int(args.start_e*num_epochs):
                data.x = torch.reshape(nodevote, (-1,1)).to(device)
            else:
                data.x = torch.ones((data.batch.shape[0], 1)).to(device)

        drop_feature = drop_feature_guided
        x_1, x_2 = drop_feature(data.x, nodevote, node_p=node_p, lambda_node= lambda_node)   
        z1 = model(x_1, edge_index_1, data.batch)
        z2 = model(x_2, edge_index_2, data.batch)
        
        loss = model.loss(z1, z2, estimator, epoch)
        
        loss.backward()
        optimizer.step()

    return loss.item()

def get_expl(model,data):
    
    if data.x is None: 
        data.x = torch.ones((data.batch.shape[0], 1)).to(device)
    nodevote = model.get_emb_avg(data.x, data.edge_index, data.batch, k_near)
    with torch.no_grad():
        nodevote = nodevote - nodevote.min()
        nodevote = nodevote / nodevote.max()
        nodevote_list = nodevote.tolist()
        
        edge = data.edge_index.tolist()
        vote = [nodevote_list[x]+nodevote_list[y] for x, y  in zip(*edge)]
        vote = torch.tensor(vote).to(device)
        vote = vote - vote.min()
        vote = vote / vote.max()
    return vote, nodevote

def test(model: Model, dataloader):
    model.eval()

    embeds = []
    labels = []

    for data in dataloader:

        data = data.to(device)
        embeddings = model(data.x, data.edge_index, data.batch)
        labels.append(data.y.detach().cpu().numpy())
        embeds.append(embeddings.detach().cpu().numpy())

    embeds = np.concatenate(embeds, 0)
    labels = np.concatenate(labels, 0)

    return mlp_evaluator(embeds, labels)

def gridtrain(data_loader, data_loader_eval):
    encoder = model_encoder(dataset.num_features, num_hidden, num_layers)
    model = Model(encoder, num_hidden, num_mid_hidden, num_layers, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_gnn, weight_decay=weight_decay)

    auEst = AUL.AUEstimator(num_hidden, learning_rate_aul, weight_decay, args.start_w*num_epochs, args.reward)
    
    start = t()
    prev = start

    for epoch in tqdm(range(1, num_epochs + 1)):
        utils.adjust_learning_rate(optimizer, epoch, learning_rate_gnn)
        loss = train(model, optimizer, data_loader, auEst, epoch)
        
        # now = t()
        # print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, 'f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        # prev = now
        # if epoch % 10 == 0:
        #     acc, acc_std, f1ma, f1ma_std, auc, auc_std = test(model, data_loader_eval)
        #     print( f'ACC: {acc:.4f}±{acc_std:.4f}, F1Ma: {f1ma:.4f}±{f1ma_std:.4f}, AUC: {auc:.4f}±{auc_std:.4f}')


    print(f"=== Result===")
    acc, acc_std, f1ma, f1ma_std, auc, auc_std = test(model, data_loader_eval)
    print( f'ACC: {acc:.4f}±{acc_std:.4f}, F1Ma: {f1ma:.4f}±{f1ma_std:.4f}, AUC: {auc:.4f}±{auc_std:.4f}')
    return acc, f1ma, auc


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='HIV')
    parser.add_argument('--config', type=str, default='BraGCL/config.yaml')
    parser.add_argument('--model', type = str, default='GCN')
    parser.add_argument('--num_gpu',type = str, default = 'cuda')
    parser.add_argument('--runtimes',type = int, default = 1)
    parser.add_argument('--start_e', type=float, default =0.3)
    parser.add_argument('--start_w', type=float, default =0.3)
    parser.add_argument('--reward', type=float, default =1)
    parser.add_argument('--modality', type=str, default='fmri')
    parser.add_argument('--seed', type=int, default=924)
    
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset][args.model]
    
    weight_decay = 0.00001
    learning_rate_gnn = config['learning_rate_gnn']
    learning_rate_aul = config['learning_rate_aul']
    num_hidden = config['num_hidden']
    num_layers = config['num_layers']
    num_mid_hidden = config['num_mid_hidden']
    num_epochs = config['num_epochs']
    model_encoder =  ({'GCN':EncoderGCN})[args.model]
    tau = config['tau']
    lambda_edge = config['lambda_edge']
    lambda_node = config['lambda_node']
    edge_p = config['edge_p']
    node_p = config['node_p']
    k_near = config['k_near']
    batchSize = config['batch_size']

    setup_seed(args.seed)
    
    device = torch.device(args.num_gpu if torch.cuda.is_available() else 'cpu')

    dataset = dataloader.MyOwnDataset('data', args.dataset, args.modality)
    

    AUC = []
    F1Ma = []
    accuracies = []
    for i in range(args.runtimes):

        dataset.shuffle()
        data_loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        data_loader_eval = DataLoader(dataset, batch_size=batchSize, shuffle=True)

        acc, f1ma, auc = gridtrain(data_loader, data_loader_eval)
        accuracies.append(acc)
        F1Ma.append(f1ma)
        AUC.append(auc)
    print(f"Final Result:\n F1Ma={np.mean(F1Ma):.4f}±{np.std(F1Ma)},\n Accuracy={np.mean(accuracies):.4f}±{np.std(accuracies)} \n AUC:{np.mean(AUC):.4f}±{np.std(AUC)}\n")
     
       

    


import Classifier
import torch
import dataloader
import torch.nn as nn
import numpy as np
import utils

from sklearn import cluster
from torch.utils.data import DataLoader
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AUEstimator:
    def __init__(self, in_channels, learning_rate, weight_decay, wtrain_epoch, reward, k=2):
        super(AUEstimator, self).__init__()
        self.learning_rate = learning_rate
        self.model = Classifier.WeightMLP(in_channels, num_classes=k+1).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.loss_function = nn.CrossEntropyLoss()
        self.wtrain_epoch = wtrain_epoch
        self.reward = reward
        self.k = k
    
    def weightCal(self, anchors, negSample, epoch):
        uncertainties = []
        for anchor in anchors:
            pLabels = torch.Tensor(binaryPartition(anchor, negSample, self.k).labels_).type(torch.LongTensor).to(device)
            
            uncertainties.append(self.uncertaintyEstimator(torch.Tensor(negSample).to(device), pLabels, epoch))
        return torch.Tensor(uncertainties)

    
    def uncertaintyEstimator(self, negSample, pLabels, epoch):

        data_loader = DataLoader(dataloader.MLPDataset(negSample, pLabels), batch_size=16, shuffle=True)
        if epoch < self.wtrain_epoch:
            utils.adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
            for _ in range(10):
                train(self.model, self.optimizer, self.loss_function, data_loader, epoch, self.reward)

        return test(self.model, data_loader)

def binaryPartition(anchor, negSample, k=2):
    k_means = cluster.KMeans(k, init=kmeans_plus(negSample, 2, anchor), n_init=1)
    k_means.fit(negSample)
    return k_means
    # return k_means

def kmeans_plus(X, n_clusters, fc):
    centroids = [fc]
    
    for _ in range(1, n_clusters):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in X])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        centroids.append(X[i])
    return np.array(centroids)


def train(model, optimizer, loss_function, data_loader, epoch, reward):

    model.train()

    for data in data_loader:
        outputs = model(data[0])
        
        if epoch >= 20:
            outputs = F.softmax(outputs, dim=1)
            outputs, reservation = outputs[:,:-1], outputs[:,-1]
            gain = torch.gather(outputs, dim=1, index=data[1].unsqueeze(1)).squeeze()
            doubling_rate = (gain.add(reservation.div(reward))).log()

            loss = -doubling_rate.mean()
        else:
            loss = loss_function(outputs[:,:-1], data[1])
        
        loss.backward()  
        optimizer.step()  
        optimizer.zero_grad()
    return None

def test(model, data_loader_eval):
    model.eval()
    abortion_results = []
    
    for data in data_loader_eval:  
        outputs = model(data[0])
        outputs = F.softmax(outputs, dim=1)
        outputs, reservation = outputs[:,:-1], outputs[:,-1]
        abortion_results.extend(list(reservation.detach().cpu().numpy()))
        
    return abortion_results

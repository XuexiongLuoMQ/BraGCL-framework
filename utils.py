import torch
import scipy.io
import numpy as np

def Adjacency_KNN(fc_data: np.ndarray, k=5):
    if k == 0:
        return np.copy(fc_data)
    adjacency = np.zeros(fc_data.shape)
    for subject_idx, graph in enumerate(fc_data):
        topk_idx = np.argsort(graph)[:, -1:-k-1:-1]
        for row_idx, row in enumerate(graph):
            adjacency[subject_idx, row_idx, topk_idx[row_idx]] = row[topk_idx[row_idx]]
    for adj in adjacency:
      for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
          if adj[i][j] != 0:
            adj[j][i] = adj[i][j]
    return adjacency

def Binary_adjacency(cor_adjacency: np.array):
   bi_adjacency = np.zeros(cor_adjacency.shape)
   for subject_idx, graph in enumerate(cor_adjacency):
      for row in range(graph.shape[0]):
         for col in range(graph.shape[1]):
            bi_adjacency[subject_idx][row][col] = 1 if graph[row][col] != 0 else 0
   return bi_adjacency

def load_data(root, name, modality='fmri'):
   file = scipy.io.loadmat(root)
   labels = torch.Tensor(file['label']).long().flatten()
   if name in ['HIV','BP']:
      data = file[modality].transpose(2,0,1)
   elif name == 'PPMI':
        X = file['X']
        data = np.zeros((X.shape[0], 84, 84))
        if modality == 'dti':
            model_index = 2
        else:
            model_index = int(modality)

        for (index, sample) in enumerate(X):
            data[index, :, :] = sample[0][:, :, model_index]
   else:
      data = file[modality]
   labels[labels == -1] = 0
   cor_adj = Adjacency_KNN(data)
   bi_adj = Binary_adjacency(cor_adj)

   return torch.Tensor(cor_adj), torch.Tensor(bi_adj), torch.Tensor(labels)

def getEdgeIdxAttr(adj):
   assert adj.dim() >= 2 and adj.dim() <= 3
   assert adj.size(-1) == adj.size(-2)

   index = adj.nonzero(as_tuple=True)
   edge_attr = adj[index]
   
   return torch.stack(index, dim=0), edge_attr


def adjust_learning_rate(optimizer, epoch, learning_rate, lrdec_1=0.8, lrdec_2=200):
    lr = learning_rate * (lrdec_1 ** (epoch // lrdec_2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 

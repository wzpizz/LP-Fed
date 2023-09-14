import torch
from sklearn.metrics import pairwise_kernels
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.stats import wasserstein_distance as w_distance
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA




class Optimization():
    def __init__(self, train_loader, device):
        self.train_loader = train_loader
        self.device = device

        # softmax归一化函数
        def softmax(x, dim=1):
            return F.softmax(x, dim=dim)

        # JS Divergence 函数
        def js_divergence(p, q):
            m = 0.5 * (p + q)
            return 0.5 * (F.kl_div(p, m) + F.kl_div(q, m))



    def cdw_feature_distance(self, old_model, old_classifier, new_model):
        """cosine distance weight (cdw): calculate feature distance of
           the features of a batch of data by cosine distance.
        """
        old_model=old_model.to(self.device)
        old_classifier=old_classifier.to(self.device)

        for data in self.train_loader:
            inputs, _ = data
            inputs = inputs.to(self.device)

            with torch.no_grad():
                old_out = old_classifier(old_model(inputs))
                new_model=new_model.to(self.device)
                new_out = new_model(inputs)
            A = old_out
            B = new_out
            A = torch.flatten(A, start_dim=1)
            B = torch.flatten(B, start_dim=1)
            AB = torch.cat([A, B], dim=0)
            corrcoef = torch.corrcoef(AB)
            AB_corrcoef = corrcoef[:A.shape[0], A.shape[0]:]
            distance=1-AB_corrcoef
            return torch.mean(distance)


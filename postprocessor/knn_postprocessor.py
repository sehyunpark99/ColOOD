from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10

class KNNPostprocessor(BasePostprocessor):
    def __init__(self):
        super(KNNPostprocessor, self).__init__()
        self.K = 50
        self.setup_flag = True
        self.activation_log = np.load('/home/shpark/Colood/postprocessor/activation_log.npy')
        self.index = faiss.read_index('/home/shpark/Colood/postprocessor/index.faiss')

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            activation_log = []
            net.eval()
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data = batch['data'].cuda()
                    data = data.float()

                    _, feature = net(data, return_feature=True)
                    activation_log.append(
                        normalizer(feature.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            print(f'KNN Activation Log: {self.activation_log}')
            self.index = faiss.IndexFlatL2(feature.shape[1])
            print(f'KNN Index: {self.index}')
            self.index.add(self.activation_log)
            # Save the index to a file
            faiss.write_index(self.index, 'index.faiss')
            self.setup_flag = True
        else:
            self.activation_log = np.load('/home/shpark/Colood/postprocessor/activation_log.npy')
            self.index = faiss.read_index('/home/shpark/Colood/postprocessor/index.faiss')
            print('Loaded activation log and index from files.')
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output, feature = net(data, return_feature=True)
        feature_normed = normalizer(feature.data.cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            self.K,
        )
        kth_dist = -D[:, -1]
        score, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return pred, torch.from_numpy(kth_dist), score

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K

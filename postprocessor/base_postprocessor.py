from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BasePostprocessor:
    def __init__(self):
        pass

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf, conf # last argument is to compare with OOD scores 

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list, score_list = [], [], [], []
        # print(f'Data_loder: {data_loader}')
        for batch in tqdm(data_loader,
                          disable=not progress): # for ood should be under
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf, score = self.postprocess(net, data)
            score = torch.tensor(score)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())
            score_list.append(score.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).detach().numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        score_list = torch.cat(score_list).detach().numpy()

        return pred_list, conf_list, label_list, score_list

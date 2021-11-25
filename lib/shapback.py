"""
TODO - docs
"""
import numpy as np

import shap
import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader


def reduce_shap(contrib, h=1, device=torch.device('cpu')):
    """TODO - docs
    Here apply the max function (Lineal instance)
    """
    if True:  # lineal instance
        shap_coeff_s = torch.tensor(1 - h*np.min(contrib, axis=1),
                                    requires_grad=True, dtype=torch.float32, device=device)
    return shap_coeff_s


def kg_from_dict(dic_s, feat, style_s):
    """TODO - docs"""
    knowledge_graph = -np.ones((len(feat), len(style_s)))
    for i in range(len(dic_s)):
        local_el = feat[i]
        for k in range(len(list(dic_s))):
            sty = list(dic_s)[k]
            if local_el in dic_s[sty]:
                knowledge_graph[i, k] = 1
    return knowledge_graph


class ShapBackLoss(nn.CrossEntropyLoss):
    """TODO - docs"""
    def __init__(self, dataset, data_train, classificator, device=torch.device('cpu')):
        super(ShapBackLoss, self).__init__(reduction='none')
        self.device = device
        self.data_train = dataset
        # self.reduction =
        self.knowledge = kg_from_dict(dataset.cmap, dataset.part_list, dataset.class_list)
        self.explainer = shap.DeepExplainer(classificator, data_train)
        # TODO - Solve errors with shap.KernelExplainer (May be TF2 exclusive)

    def compare_shap_and_kg(self, shap_values, true_labels, features, threshold=0):
        """TODO - docs
        Functions to compute instance level weights. Here compute misattribution
        """
        contrib = np.zeros((len(true_labels), len(features)))
        for k, tl in enumerate(true_labels):
            local_kg = self.knowledge[:, tl]
            contrib[k] = shap_values[tl][k] * local_kg
        contrib[contrib > -threshold] = 0
        return contrib

    def forward(self, y_pred: Tensor, target: Tensor, elems: Tensor = None) -> Tensor:
        target, elems = target.long(), elems.float()

        # SHAP
        if elems is not None and torch.is_grad_enabled():
            shap_values = self.explainer.shap_values(elems)
            shap_contrib = self.compare_shap_and_kg(shap_values, target, self.data_train.part_list)
            shap_coeff = reduce_shap(shap_contrib, device=self.device)
            # TODO - GED metric
        else:
            shap_coeff = torch.ones((len(y_pred), ), device=self.device)

        loss = super(ShapBackLoss, self).forward(y_pred, target)
        # if self.reduction == 'none': ...
        return torch.dot(loss, shap_coeff)/len(y_pred)


def test():
    import os
    from utils.load_data import get_dataset

    os.chdir('..')
    torch.manual_seed(621)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_train, data_tests = get_dataset('FFoCat', device=dev)
    model = nn.Sequential(
        nn.Linear(in_features=len(data_train.part_list), out_features=11), nn.ReLU(),
        nn.Linear(in_features=11, out_features=len(data_train.class_list))  # , nn.Softmax(dim=-1),
    );  model.to(dev)
    loss = ShapBackLoss(dataset=data_train.classify, data_train=data_train.classify[:][1][0],
                        classificator=model, device=dev)
    opt = torch.optim.Adam(model.parameters())

    train_data = DataLoader(data_train.classify, batch_size=8, shuffle=True)
    _, (X_batch, Y_batch) = next(iter(train_data))

    model.train()
    print(next(iter(model.parameters())).data[0])
    yb_pred = model(X_batch)
    batch_loss = loss(yb_pred, Y_batch, X_batch)
    batch_loss.backward()
    opt.step()
    print(next(iter(model.parameters())).data[0])


if __name__ == '__main__':
    test()

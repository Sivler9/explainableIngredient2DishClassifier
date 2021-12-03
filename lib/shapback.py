"""
TODO - docs
"""
import numpy as np

import shap
import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader

from utils.load_data import ShapImageDataset


def reduce_shap(contrib, h=1, device=torch.device('cpu')):
    """TODO - docs
    Here apply the max function (Lineal instance)
    """
    if True:  # lineal instance
        shap_coeff_s = torch.tensor(1 - h * np.min(contrib, axis=1), requires_grad=True,
                                    dtype=torch.float, device=device)
    return shap_coeff_s


def kg_from_dict(dic_s, feat=None):
    """TODO - docs"""
    if feat is None:
        feat = sorted(set([el for sty in dic_s.values() for el in sty]))
    knowledge_graph = -np.ones((len(feat), len(dic_s)))
    for i, local_el in enumerate(feat):
        for k, sty in enumerate(sorted(dic_s.keys())):
            if local_el in dic_s[sty]:
                knowledge_graph[i, k] = 1
    return knowledge_graph


def graph_edit_distance(features, shap_values, dataset: ShapImageDataset, threshold=0.001):
    """TODO - docs
    Fully compute the GED, given feature vectors and shap values. Build KG, build SAG and compute GED"""
    def build_kg(dictionary):
        graph = {}
        for k, v in dictionary.items():
            k, v = f'c{clases.index(k)}', set([f'p{parts.index(n)}' for n in v])
            graph[k] = v
            for n in v:
                if n not in graph:
                    graph[n] = {k}
                else:
                    graph[n].add(k)
        return graph

    def graph_distance(expert, sag):  # div 2 - because graph is not directed
        return sum([len(expert[n] ^ sag[n]) for n in expert.keys() & sag.keys()])/2.

    KG, parts, clases = dataset.cmap, dataset.part_list, dataset.class_list
    features = features.detach().cpu().numpy()
    shap_array = np.dstack(shap_values)
    features[features < .1] = 0
    expert_graph = build_kg(KG)

    d_tot = 0
    for i in range(len(shap_array)):
        shap_graph = {}
        for k in range(shap_array.shape[-1]):
            facade = np.copy(shap_array[i, :, k])
            for j, f in enumerate(features[k] > 0.):
                clas, part = f'c{k}', f'p{j}'
                if not f ^ (facade[j] > threshold):
                    if clas not in shap_graph:
                        shap_graph[clas] = {part}
                    else:
                        shap_graph[clas].add(part)
                    if part not in shap_graph:
                        shap_graph[part] = {clas}
                    else:
                        shap_graph[part].add(clas)
            # TODO - Report bug to original repo
        d_tot += graph_distance(expert_graph, shap_graph)
    return float(d_tot) / len(shap_array)


class ShapBackLoss(nn.CrossEntropyLoss):
    """TODO - docs"""

    def __init__(self, dataset: ShapImageDataset, data_train, classificator, device=torch.device('cpu')):
        super(ShapBackLoss, self).__init__(reduction='none')  # self.reduction =
        self.dataset, self.device = dataset, device
        self.explainer = shap.DeepExplainer(classificator, data_train)
        # TODO - Solve errors with shap.KernelExplainer (May be TF2 exclusive)
        self.knowledge = kg_from_dict(dataset.cmap, dataset.part_list)

    def compare_shap_and_kg(self, shap_values, true_labels, features, threshold=0):
        """TODO - docs
        Functions to compute instance level weights. Here compute misattribution.
        """
        contrib = np.zeros((len(true_labels), len(features)))
        for k, tl in enumerate(true_labels):
            local_kg = self.knowledge[:, tl]
            contrib[k] = shap_values[tl][k] * local_kg
        contrib[contrib > -threshold] = 0
        return contrib

    def shap_coefficient(self, y_pred: Tensor, target: Tensor, elems: Tensor = None):
        target, elems = target.long(), elems.float()
        # SHAP
        if elems is not None and torch.is_grad_enabled():
            shap_values = self.explainer.shap_values(elems)
            shap_contrib = self.compare_shap_and_kg(shap_values, target, self.dataset.part_list)
            shap_coeff = reduce_shap(shap_contrib, device=self.device)
            ged = graph_edit_distance(elems, shap_values, self.dataset, threshold=.001)
        else:
            shap_coeff = torch.ones((len(y_pred),), device=self.device)
            ged = np.nan

        # if self.reduction == 'none': ...
        # torch.dot(loss, shap_coeff) / len(y_pred), ged
        # if ged is np.nan else (torch.dot(loss, shap_coeff) / len(y_pred), ged)
        return shap_coeff, ged


def test():
    import os
    from utils.load_data import get_dataset

    os.chdir('..')
    torch.manual_seed(621)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_train, data_tests = get_dataset('FFoCat_tiny', device=dev)
    model = nn.Sequential(
        nn.Linear(in_features=len(data_train.part_list), out_features=11), nn.ReLU(),
        nn.Linear(in_features=11, out_features=len(data_train.class_list)),  # nn.Softmax(dim=-1),
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

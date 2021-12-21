"""
TODO - docs and type hints
"""
import numpy as np

import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader

from utils.load_data import ShapImageDataset

import warnings
with warnings.catch_warnings():  # (record=True) as w:
    warnings.simplefilter('ignore')
    import shap


def kg_from_dict(dic_s, feat=None):
    """
    Create knowledge graph from class map

    :param dict dic_s: Class tp feature map
    :param list feat: Pre-computed features list
    :returns np.ndarray: Knowledge Graph
    """
    if feat is None:
        feat = sorted(set([el for sty in dic_s.values() for el in sty]))
    knowledge_graph = -np.ones((len(feat), len(dic_s)), dtype=np.uint8)
    for i, local_el in enumerate(feat):
        for k, sty in enumerate(sorted(dic_s.keys())):
            if local_el in dic_s[sty]:
                knowledge_graph[i, k] = 1
    return knowledge_graph


def shap_graph_edit_distances(features, shap_values, dataset: ShapImageDataset, threshold=0.001):
    """Fully compute the GED, given feature vectors and shap values. Build KG, build SAG and compute GED
    TODO - docs
    :param features:
    :param shap_values:
    :param dataset:
    :param threshold:
    :return:
    """
    def build_kg(dictionary, classes, parts):
        graph = {}
        for k, v in dictionary.items():
            k, v = f'c{classes.index(k)}', set([f'p{parts.index(n)}' for n in v])
            graph[k] = v
            for n in v:
                if n not in graph:
                    graph[n] = {k}
                else:
                    graph[n].add(k)
        return graph

    def filtered_edit_distance(expert, sag):  # div 2 - because graph is not directed
        sag_k = expert.keys() & sag.keys()  # == sag.keys() (ideally, anyways)
        dist = sum([len((expert[n] ^ sag[n]) & sag_k) for n in sag_k])/2.
        assert int(dist) == dist
        return dist

    expert_graph = build_kg(dataset.cmap, dataset.class_list, dataset.part_list)
    features = features.detach().cpu().numpy()
    shap_array = np.dstack(shap_values)
    features[features < .5] = 0.

    d_tot = []
    for i in range(len(shap_array)):
        shap_graph = {}
        feats = features[i]
        feats_too = feats + (feats == 0.)
        for k in range(shap_array.shape[-1]):
            facade = shap_array[i, :, k]*feats_too
            for j, f in enumerate(feats > 0.):  # Always all positive or 0
                if not f:  # TODO - bug in original code - was always False (changes the results a lot)
                    continue
                clas, part = f'c{k}', f'p{j}'
                # XOR - /either/ feat*shap  > threshold   when   feats is /not/  > 0
                #          /xor/ feat*shap <= threshold   when   feats is /not/ <= 0
                if not f ^ (facade[j] > threshold):
                    if clas not in shap_graph:
                        shap_graph[clas] = {part}
                    else:
                        shap_graph[clas].add(part)
                    if part not in shap_graph:
                        shap_graph[part] = {clas}
                    else:
                        shap_graph[part].add(clas)
        d_tot.append(filtered_edit_distance(expert_graph, shap_graph))
    return d_tot  # float(d_tot) / shap_array.shape[0]


def compare_shap_and_kg(knowledge, shap_values, true_labels, threshold=0.):
    """Computes misattribution.
    TODO - docs
    :param shap_values:
    :param true_labels:
    :param features:
    :param threshold:
    :return:
    """
    contrib = np.zeros(shap_values[0].shape)  # (len(true_labels), len(features))
    for k, tl in enumerate(true_labels.view(-1, 1)):
        local_kg = knowledge[:, tl]
        contrib[k] = shap_values[tl][k] * local_kg
    contrib[contrib > -threshold] = 0.
    return contrib


def reduce_shap(contrib, h=1, exp=False, device=torch.device('cpu')):
    """Apply the max function
    TODO - docs
    :param contrib:
    :param h:
    :param exp:
    :param device:
    :return:
    """
    shap_coeff_s = -h*np.min(contrib, axis=1)
    if exp:  # exponential
        shap_coeff_s = np.exp(shap_coeff_s)
    else:  # lineal instance
        shap_coeff_s = 1 + shap_coeff_s
    return torch.tensor(shap_coeff_s, requires_grad=False, dtype=torch.float, device=device)


class ShapBackLoss(nn.CrossEntropyLoss):
    """TODO - docs"""

    def __init__(self, dataset: ShapImageDataset, data_train, classificator, device=torch.device('cpu')):
        super(ShapBackLoss, self).__init__()  # (reduction='none')  # self.reduction =
        self.dataset, self.device = dataset, device
        self.explainer = shap.DeepExplainer(classificator, data_train.to(device))
        # TODO - Solve errors with shap.KernelExplainer (May be TF2 exclusive)
        self.knowledge = kg_from_dict(dataset.cmap, dataset.part_list)

    def train_classifier(self):  # TODO - Move classifier training to here
        raise NotImplementedError()

    def shap_coefficient(self, y_pred: Tensor, target: Tensor, elems: Tensor = None):
        target, elems = target.long(), elems.float()
        # SHAP
        if elems is not None:  # and torch.is_grad_enabled():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                shap_values = self.explainer.shap_values(elems)  # needs grad_enabled, it seems
            shap_contrib = compare_shap_and_kg(self.knowledge, shap_values, target, threshold=.0)
            shap_coeff = reduce_shap(shap_contrib, device=self.device)
            ged = shap_graph_edit_distances(elems, shap_values, self.dataset, threshold=.001)
        else:
            shap_coeff = torch.ones((len(y_pred),), device=self.device)
            ged = np.empty((len(y_pred),))
            ged[:] = np.nan

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
    loss = ShapBackLoss(dataset=data_train, data_train=data_train[:16][1][1],
                        classificator=model, device=dev)
    opt = torch.optim.Adam(model.parameters())

    train_data = DataLoader(data_train.classify, batch_size=8, shuffle=True)
    _, (_, X_batch, Y_batch) = next(iter(train_data))

    model.train()
    print(next(iter(model.parameters())).data[0])
    yb_pred = model(X_batch)
    batch_loss = loss(yb_pred, Y_batch, X_batch)
    batch_loss.backward()
    opt.step()
    print(next(iter(model.parameters())).data[0])


if __name__ == '__main__':
    test()

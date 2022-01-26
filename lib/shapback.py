"""
TODO - docs and type hints
"""

import torch
import numpy as np
import torch.nn as nn

from tqdm import trange
from torch import Tensor
from torch.utils.data import DataLoader

from .utils.metrics import categorical_accuracy
from .utils.knowledge_graph import kg_matrix, kg_build
from .utils.load_data import ShapImageDataset, move_to_device

import warnings
with warnings.catch_warnings():  # (record=True) as w:
    warnings.simplefilter('ignore', UserWarning)  # IPython
    import shap


def compute_sag(feature_vector, shap_values, dataset, threshold=.1):
    """Implements 'Algorithm 1' in 'EXplainable Neural-Symbolic Learning (X-NeSyL) methodology to fuse deep learning
     representations with expert knowledge graphs: the MonuMAI cultural heritage use case'
     Code from

    :param feature_vector:
    :param shap_values:
    :param dataset:
    :param threshold:
    :return:
    """
    sag = {}
    for k, c in enumerate(dataset.class_list):
        local_shap = shap_values[:, k]
        sag[c] = set()
        for j, p in enumerate(dataset.part_list):
            shap_val = local_shap[j]
            if feature_vector[j] > threshold:
                if shap_val > .0:
                    sag[c].add(p)
            elif shap_val < .0:
                sag[c].add(p)
    return kg_build(sag, dataset.class_list, dataset.part_list)


def filtered_edit_distance(expert: dict, sag: dict):
    sag_k = expert.keys() & sag.keys()  # == sag.keys() - ideally, anyways, otherwise there is something really wrong
    dist = sum([len((expert[n] ^ sag[n]) & sag_k) for n in sag_k])/2.  # div 2 - because graph is not directed
    assert int(dist) == dist
    return dist


def shap_graph_edit_distances(features, shap_values, dataset: ShapImageDataset, threshold=.5):
    """Fully compute the GED, given feature vectors and shap values. Build KG, build SAG and compute GED
    TODO - docs
    :param torch.Tensor features:
    :param np.ndarray shap_values:
    :param ShapImageDataset dataset:
    :param float threshold:
    :returns list:
    """
    expert_graph = kg_build(dataset.cmap, dataset.class_list, dataset.part_list)
    features = features.detach().cpu().numpy()
    shap_array = np.dstack(shap_values)
    features[features < threshold] = 0.

    d_tot = []
    for i in range(len(shap_array)):
        shap_graph = compute_sag(features[i], shap_array[i], dataset)
        d_tot.append(filtered_edit_distance(expert_graph, shap_graph))
    return d_tot  # float(d_tot) / shap_array.shape[0]


def compare_shap_and_kg(knowledge: np.ndarray, shap_values: np.ndarray, true_labels: torch.Tensor, threshold=0.):
    """Computes misattribution

    :param np.ndarray knowledge: Adjacency matrix
    :param np.ndarray shap_values: Output from shap.shap_values(...)
    :param torch.Tensor true_labels: Ground truth (sparse)
    :param float threshold: Minimum value to contibute to graph
    :returns np.ndarray:
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


class ShapBackLoss(nn.BCEWithLogitsLoss):
    def __init__(self, data: DataLoader, neurons: int = None, classifier=None,
                 criterion=None, optimizer=None, scheduler=None, metric=None,
                 reduction='mean', device=torch.device('cpu')):
        super(ShapBackLoss, self).__init__(reduction='none')
        dataset = data.dataset
        assert isinstance(dataset, ShapImageDataset)
        if neurons is None:
            neurons = min(3*len(dataset.class_list) - 1, (len(dataset.class_list) + len(dataset.part_list))//2 + 2)
        self.knowledge = kg_matrix(dataset.cmap, dataset.part_list, dataset.class_list)
        self.classifier = classifier if classifier is not None else nn.Sequential(
            nn.Linear(len(dataset.part_list), neurons), nn.ReLU(inplace=True),
            # Optional (not in original) - does not increase exec time noticeably
            # nn.Linear(neurons, neurons), nn.ReLU(inplace=True),
            nn.Linear(neurons, len(dataset.class_list)),  # nn.Softmax(dim=-1),
        ).to(device)

        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam
        self.metric = metric if metric is not None else categorical_accuracy
        self.initial_classifier = None
        self.scheduler = scheduler
        self.reduce = reduction
        self.dataset = dataset
        self.explainer = None

        self.previous_inputs = torch.zeros(len(dataset), len(dataset.part_list), device=device)
        self.previous_truth = torch.zeros(len(dataset), dtype=torch.long, device=device)
        self.shap_coeffs = torch.ones(len(dataset), device=device)
        self.ids = 0

        self._init_train(data)

    @property
    def device(self):
        return next(self.classifier.parameters()).device

    def to(self, device, *args, **kwargs):  # TODO - Others
        return self.classifier.to(device, *args, **kwargs)

    def shap_coefficients(self, max_samples=16, input_size=None):
        if input_size and input_size != self.previous_truth.size(0):
            elems = self.previous_inputs[:input_size].float()
            target = self.previous_truth[:input_size].long()
        else:
            elems = self.previous_inputs.float()
            target = self.previous_truth.long()
        num_samples = min(target.size(0), max_samples)
        if False and target.size(0) < 1000:  # TODO - option
            samples = elems[np.random.choice(target.size(0), num_samples, False)]
        else:
            print('Summarizing dataset with kmeans...', end='\r')
            samples = torch.tensor(shap.kmeans(elems.cpu(), num_samples).data, dtype=torch.float, device=self.device)
        # SHAP
        if torch.is_grad_enabled():
            explainer = shap.DeepExplainer(self.classifier, samples)
            with warnings.catch_warnings():
                print('Getting shap values...', end='\r')
                warnings.simplefilter("ignore", UserWarning)
                shap_values = explainer.shap_values(elems)  # needs grad_enabled, it seems
            ged = shap_graph_edit_distances(elems, shap_values, self.dataset, threshold=.5)
            if shap_values[0].shape[0] == self.previous_truth.size(0):
                shap_contrib = compare_shap_and_kg(self.knowledge, shap_values, target, threshold=.0)
                self.shap_coeffs = reduce_shap(shap_contrib, device=self.device).detach()
        else:
            ged = np.empty_like(self.shap_coeffs)
            ged[:] = np.nan()
        return np.mean(ged).item()

    def train_epoch(self, optimizer, batch=32, input_size=None):
        length = self.previous_inputs.size(0) if input_size is None else input_size
        length, loss, acc = length//batch, 0., 0.
        for ran in range(length):
            idx = slice(batch*ran, batch*(ran + 1))
            inp = self.previous_inputs[idx]
            tru = self.previous_truth[idx]
            # Network & Loss output
            res = self.classifier(inp)
            tmp_loss = self.criterion(res, tru)
            # Optimizer
            optimizer.zero_grad()
            tmp_loss.backward()
            optimizer.step()
            # Log
            loss += tmp_loss.item()
            acc += torch.mean(self.metric(res, tru))
        return loss/length, acc/length

    def step(self, epochs=25, train=True, shap_train=True, input_size=None):
        self.ids = 0
        optimizer = self.optimizer(self.classifier.parameters())
        scheduler = self.sheduler(optimizer) if self.optimizer is not torch.optim.Adam else None
        if self.initial_classifier:
            self.classifier.load_state_dict(self.initial_classifier)
        loss, acc = 0., 0.
        for ep in trange(epochs if train else 1, desc='Shap Epochs', disable=not train):
            loss, acc = self.train_epoch(optimizer)
            if scheduler:
                scheduler.step()
        opt = {} if input_size is None else {'input_size': input_size}
        ged = self.shap_coefficients(**opt) if shap_train else np.nan
        return {'shap_loss': loss, 'shap_acc': acc, 'shap_ged': ged}

    def _init_train(self, data, epochs=5):
        for dat in data:
            imgs, (targets, parts, clases) = move_to_device(dat, device=self.device)
            self.previous_inputs[[tar['image_id'] for tar in targets], :] = parts
        self.step(epochs, train=True, shap_train=False)
        self.initial_classifier = self.classifier.state_dict()

    def forward(self, predicted,  ground_truth, macro_labels: torch.Tensor = None, ids: list = None):  #  -> torch.Tensor:
        if ids is None:  # TODO - based on batches
            tmp = self.ids + predicted.size(0)
            ids = slice(self.ids, tmp)
            self.ids = tmp
        self.previous_inputs[ids, :] = predicted.detach()
        if macro_labels is not None:
            self.previous_truth[ids] = macro_labels.detach()
        loss = torch.mean(super(ShapBackLoss, self).forward(predicted, ground_truth), dim=-1)
        if self.reduce == 'none':
            return torch.dot(loss, self.shap_coeffs[ids])
        elif self.reduce == 'split':
            return loss, self.shap_coeffs[ids]
        elif self.reduce == 'mean':
            return torch.mean(torch.dot(loss, self.shap_coeffs[ids]))
        else:  # self.reduce == 'sum':
            return torch.sum(torch.dot(loss, self.shap_coeffs[ids]))


def tests():  # TODO - move tests to their own directory
    raise NotImplementedError


if __name__ == '__main__':
    tests()


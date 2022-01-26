"""
TODO - docs
"""
import torch
import numpy as np

try:
    from metrics_tests import *
except ImportError as e:
    pass


class History:
    def __init__(self):
        self.log = {}

    def update(self, dic: dict):
        length = len(next(iter(self.log.values()))) if self.log else 0
        for key, val in dic.items():
            val = [val.item() if torch.is_tensor(val) else val]
            self.log[key] = (self.log[key] if self.log.get(key) else [np.nan]*length) + val

    def __getitem__(self, item):
        return self.log.__getitem__(item)


def categorical_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor):
    return torch.eq(y_true, torch.argmax(y_pred.detach(), dim=-1)).float()


def binary_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=torch.FloatTensor([.5])):
    return torch.mean(torch.eq(torch.gt(y_true, 0.), torch.gt(y_pred.detach(), threshold.to(y_true.device))).float(),
                      dim=-1)


def tests():  # TODO - move tests to their own directory
    raise NotImplementedError


if __name__ == '__main__':
    tests()

import gc
import os
import sys
import time
import random
import pickle

import torch.optim
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from lib.shapback import ShapBackLoss
from lib.utils.metrics import History, binary_accuracy
from lib.utils.load_data import get_dataset, shap_collate_fn, move_to_device

# Typing
from typing import Union
from lib.utils.load_data import ShapImageDataset

DEBUGGING = sys.gettrace() is not None
SAVE_DIR = f'./results/{int(time.time())}/'
DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def box_scores_to_array(x: list[dict], num_parts: int, device=torch.device('cpu')) -> torch.Tensor:
    """Transforms dictionary list with object detections results into a multilabel equivalent output

    :param list[dict] x: Output from a `torchvision.models.detection.generalized_rcnn.GeneralizedRCNN` subclass
    :param int num_parts: Number of labels for the multilabel classificator
    :param Optional[torch.device] device: Output device (default: cpu)
    :returns torch.Tensor: Aggregate scores per part
    """
    aggs = []
    for d in x:
        agg = [0.] * num_parts
        for i, l in enumerate(d['labels'].detach().cpu().numpy()):
            agg[l] += d['scores'][i].item()
        aggs.append(torch.tensor(agg, dtype=torch.float))
    return torch.stack(aggs).to(device)


def adjust_output_size(model, out_size, postprocess=tuple()):
    """Modifies last layer of a pytorch model

    :param nn.Module model: The model
    :param out_size: Number of out features
    :param Union[list, tuple] postprocess: ej.: [nn.Sigmoid()]
    :returns: nn.Module
    """
    if hasattr(model, 'roi_heads'):  # models.detection.FasterRCNN
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, out_size)
        return False
    elif len(list(model.children())):  # TODO - Detect if the same model has passed before to prevent nesting Sequential
        for name, lay in reversed(list(model.named_children())):
            if 'Linear' in lay.__repr__():
                if adjust_output_size(lay, out_size):
                    if lay.out_features == out_size:
                        if postprocess:
                            setattr(model, name, nn.Sequential(lay, *postprocess))
                    else:
                        if postprocess:
                            setattr(model, name, nn.Sequential(nn.Linear(lay.in_features, out_size), *postprocess))
                        else:
                            setattr(model, name, nn.Linear(lay.in_features, out_size))
                break  # return False
        else:
            raise Warning('Cannot find layer to replace')
    else:
        return True  # TODO - Probaly do something (or skip) if the entire model is only a layer
    return False


def eager_outputs(losses, detections):
    return losses, detections


class DetectorWrapper(nn.Module):
    def __init__(self, model: nn.Module, dataset: ShapImageDataset,
                 criterion=None, optimizer=None, scheduler=None, metric=None,
                 criterion_opt=None, optimizer_opt=None, scheduler_opt=None, box_convert='raw', threshold=.5):
        super(DetectorWrapper, self).__init__()
        self.model, self.threshold, self.scheduler = model, threshold, scheduler
        if criterion_opt is None:
            criterion_opt = dict()
        if optimizer_opt is None:
            optimizer_opt = dict()
        if scheduler_opt is None:
            scheduler_opt = dict()
        # TODO - hyperparameters
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), **optimizer_opt)
        self.criteria = criterion if criterion is not None else nn.BCEWithLogitsLoss(**criterion_opt)
        self.metric = metric if metric is not None else binary_accuracy
        self.scheduler = scheduler if scheduler is not None else None

        self.box = isinstance(model, models.detection.generalized_rcnn.GeneralizedRCNN)
        if self.box:
            # TODO - output losses and detection in the same forward()
            # self.model = torch.jit.trace(self.model, torch.rand(1, 3, 224, 224))
            model.eager_outputs = eager_outputs
        if self.box and not callable(box_convert):  # So we dont need to compute it twice
            if box_convert == 'detections':
                self.detector_output_translation = lambda x: torch.gt(
                    box_scores_to_array(x, len(dataset.part_list), self.device),
                    self.detection_threshold).float()
            else:
                self.detector_output_translation = lambda x: \
                    box_scores_to_array(x, len(dataset.part_list), self.device)
        if callable(box_convert):
            self.detector_output_translation = box_convert
        elif box_convert == 'detections':
            self.detector_output_translation = lambda x: torch.gt(x.detach(), self.detection_threshold).float()
        elif not self.box:  # Raw output
            self.detector_output_translation = lambda x: x.detach().float()

        self.previous_output = None
        self.previous_truth = None
        self.previous_ids = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def to(self, *args, **kwargs):  # device, dtype, non_blocking
        self.model = self.model.to(*args, **kwargs)
        return self  # TODO - other objs ...problably

    def criterion(self, predicted, ground_truth):
        if isinstance(self.criteria, ShapBackLoss):
            return self.criteria(predicted, *ground_truth, self.previous_ids)
        else:
            return self.criteria(predicted, ground_truth[0])

    def loss(self):
        assert self.previous_output is not None, 'Needs to run at least 1 .forward()'
        if self.box:
            loss = sum(po for po in self.previous_output[0].values())
        elif hasattr(self.previous_output, 'logits'):  # Inception (and apparently GoogLeNet)
            logits = self.previous_output.logits
            loss = self.criterion(logits, self.previous_truth)
            for k in self.previous_output._fields:
                aux_logits = getattr(self.previous_output, k)
                if k != 'logits' and aux_logits.size() == logits.size():
                    loss += .3 * self.criterion(aux_logits, self.previous_truth)
        else:
            loss = self.criterion(self.previous_output, self.previous_truth)
        return loss

    def accuracy(self):  # TODO
        ground_truth = self.previous_truth[0]
        if self.box:
            acc = self.detector_output_translation(self.previous_output[-1])
        elif hasattr(self.previous_output, 'logits'):  # Inception (and apparently GoogLeNet)
            acc = self.previous_output.logits
        else:
            acc = self.previous_output
        return torch.mean(self.metric(acc, ground_truth))

    def step(self):
        loss = self.loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), self.accuracy()

    def forward(self, data: Union[list[Union[torch.Tensor, list[dict]]], torch.Tensor],
                targets: list[dict] = None) -> torch.Tensor:
        if torch.is_tensor(data):
            if self.train() and targets is not None:
                self.previous_ids = [tar['image_id'] for tar in targets]
                self.previous_output = self.model(data, targets)
                self.previous_truth = targets
            else:
                self.previous_output = self.model(data)
                self.previous_truth = None
                self.previous_ids = None
        else:
            imgs, (targets, parts, clas) = move_to_device(data, device=self.device)
            self.previous_truth = parts, clas
            self.previous_ids = [tar['image_id'].item() for tar in targets]
            if self.box:
                self.model.eval()
                with torch.no_grad():
                    tmp = self.model(imgs)[-1]
                self.model.train()
                self.previous_output = self.model(imgs, targets)[0], tmp
            else:
                self.previous_output = self.model(imgs)
        y = self.previous_output[-1] if isinstance(self.previous_output, (list, tuple)) else self.previous_output
        return self.detector_output_translation(y)


def save_model(model, path):
    a = next(model.parameters()).device
    try:
        torch.save(model.cpu(), path)
    except pickle.PicklingError:
        b = model.eager_outputs
        model.eager_outputs = models.detection.generalized_rcnn.GeneralizedRCNN.eager_outputs
        torch.save(model.cpu(), path)
        model.eager_outputs = b
    model.to(a)


class EXPLANetTrainer:
    def __init__(self):
        self.history = History()

    def train_epoch(self, detector, train_data):
        detector.train()
        loss, acc = 0., 0.
        for batch in tqdm(train_data, desc='Training Batches', ncols=80):
            detector(batch)
            tmp_loss, tmp_acc = detector.step()
            loss += tmp_loss
            acc += tmp_acc
        if detector.scheduler:
            detector.scheduler.step()
        loss, acc = loss/len(train_data), acc/len(train_data)
        shap_metrics = {'train_' + k: v for k, v in detector.criteria.step(
            ).items()} if hasattr(detector.criteria, 'step') else {}
        self.history.update(dict(train_loss=loss, train_acc=acc, **shap_metrics))

    def valid_epoch(self, detector, valid_data, shap_ged=False):
        loss, acc = 0., 0.
        detector = detector.train() if detector.box else detector.eval()
        with tqdm(valid_data, desc='Validation Batches', ncols=80) as batches, torch.no_grad():
            for batch in batches:
                detector(batch)
                acc += detector.accuracy()
                loss += detector.loss().item()
        loss, acc = loss/len(valid_data), acc/len(valid_data)
        shap_metrics = {'valid_' + k: v for k, v in detector.criteria.step(train=False,
            input_size=len(valid_data.dataset)).items()} if hasattr(detector.criteria, 'step') and shap_ged else {}
        self.history.update(dict(valid_loss=loss, valid_acc=acc, **shap_metrics))

    def train_net(self, detector: DetectorWrapper, train_data: DataLoader, valid_data: DataLoader = None,
                  epochs=50, patience=10, verbose=1, name_id=''):
        assert isinstance(train_data.dataset, ShapImageDataset), 'The DaraLoader needs to contain ShapImageDataset'
        if detector.box:
            train_data.dataset.count = True
            if valid_data:
                valid_data.dataset.count = True
        best = np.inf
        epochs_since_best = 0
        print(f'[INFO] starting training (date: {time.strftime("%Y-%b-%d %H:%M:%S", time.localtime())})')
        for ep in range(epochs):
            start_time = time.time()
            self.train_epoch(detector, train_data)
            valid_1, valid_2, shap_1, shap_2 = '', '', '', ''
            if valid_data:
                epochs_since_best += 1
                self.valid_epoch(detector, valid_data)
                if self.history['valid_loss'][-1] < best:
                    epochs_since_best = 0
                    best = self.history['valid_loss'][-1]
                    save_model(detector.model, os.path.join(SAVE_DIR, 'models', f'{name_id}_d{ep}.pth'))
                    save_model(detector.criteria.classifier, os.path.join(SAVE_DIR, 'models', f'{name_id}_c{ep}.pth'))
                valid_1, valid_2 = f'|{self.history["valid_acc"][-1]:.5f}', f'|{self.history["valid_loss"][-1]:.5f}'
            if 'train_shap_acc' in self.history.log:
                shap_1 = f', cat_acc={self.history["train_shap_acc"][-1]:.5f}'
                shap_2 = f', shap_ged={self.history["train_shap_ged"][-1]:.5f}'
                if 'valid_shap_acc' in self.history.log:
                    shap_1 += f'|{self.history["valid_shap_acc"][-1]:.5f}'
                    shap_2 += f'|{self.history["valid_shap_ged"][-1]:.5f}'
            if verbose:
                print(f'\rEpoch {ep + 1:03d}/{epochs:03d} -- '
                      f'bin_acc={self.history["train_acc"][-1]:.5f}{valid_1}{shap_1}'
                      f', loss={self.history["train_loss"][-1]:.5f}{valid_2}{shap_2}'
                      f' [patience: {patience - epochs_since_best}/{patience}, time: {time.time() - start_time:.5f}s]')
            if epochs_since_best > patience:
                print('[INFO]Exceeded patience')
                break
        print(f'[INFO] finished training (date: {time.strftime("%Y-%b-%d %H:%M:%S", time.localtime())})')
        return self.history.log


def main():  # TODO - verbose for training tqdm
    # Reproducibility
    random.seed(621)
    np.random.seed(621)
    torch.manual_seed(621)

    # Set up save directory
    os.makedirs(os.path.join(SAVE_DIR, 'code'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'models'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'figures'), exist_ok=True)
    root_dir, file_name = os.path.split(os.path.realpath(__file__))
    code, lib, utils = os.path.join(root_dir, file_name), os.path.join(root_dir, 'lib'), \
        os.path.join(root_dir, 'lib', 'utils')

    # Define combinations
    databases = ['MonuMAI', 'FFoCat', 'PASCAL']  # (*_test) (FFoCat_reduced, FFoCat_tiny)
    backbones = [models.resnet50, models.mobilenet_v2, models.inception_v3, models.detection.fasterrcnn_resnet50_fpn]

    if DEBUGGING:
        os.system(f'cp -r {code} {lib} {utils} {SAVE_DIR}/code')

    # Execution
    for db in databases:
        for back in backbones:
            # Prepare model and data
            backbone = back(pretrained=True)  # aux_logits=False
            if hasattr(backbone, 'roi_heads') and 'FFoCat' in db:
                continue  # FFoCat does not have boxes
            name_id = f'{db}_{back.__name__}'  # backbone.__class__.__name__
            img_size = 299 if isinstance(backbone, models.Inception3) else 224
            dbs, batch_size, workers, pre_epochs, epochs = db, (8 if hasattr(backbone, 'roi_heads') else 32), 4, 5, 50
            if DEBUGGING:
                dbs, batch_size, workers, pre_epochs, epochs = db + '_test', 8, 0, 2, 2
            data_train, data_valid = get_dataset(dbs, size=img_size, device=torch.device('cpu'))
            adjust_output_size(backbone, len(data_train.part_list))  # Can't do num_classes if pretrained

            # Prepare training
            data_opt = dict(batch_size=batch_size, collate_fn=shap_collate_fn, num_workers=workers, pin_memory=True)
            t_data = DataLoader(data_train, **data_opt, shuffle=True)  # , worker_init_fn=lambda x: breakpoint())
            v_data = DataLoader(data_valid, **data_opt)
            detector = DetectorWrapper(backbone, data_train, criterion=nn.BCEWithLogitsLoss()).to(DEFAULT_DEVICE)

            # Run training
            trainer = EXPLANetTrainer()
            print(f'[INFO] {pre_epochs} epochs training of detector and classifier individually')
            trainer.train_net(detector, t_data, epochs=pre_epochs)
            detector.criteria = ShapBackLoss(t_data, 3*len(data_train.class_list) - 1, device=DEFAULT_DEVICE)
            print(f'\r[INFO] now training with ShapBackLoss - {name_id}')
            hist = trainer.train_net(detector, t_data, v_data, epochs=epochs, name_id=name_id)

            # Save model and results
            with open(os.path.join(SAVE_DIR, f'hist_{name_id}.pkl'), 'wb+') as pickle_file:
                pickle.dump(hist, pickle_file)

    import autoenc
    autoenc.main()


if __name__ == '__main__':
    main()

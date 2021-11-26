import time
import copy
import math
import random
import os.path

import numpy as np
import torch.optim
import torch.nn as nn
import matplotlib.pyplot as plt

from matplotlib import patches
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.load_data import *
from lib.shapback import ShapBackLoss

DEFAULT_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def categorical_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor):
    return torch.eq(y_true, torch.argmax(y_pred.detach(), dim=-1)).float()


def binary_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor,
                    threshold=torch.FloatTensor([.5]), device=DEFAULT_DEVICE):
    return torch.mean(torch.eq(y_true, torch.gt(y_pred.detach(), threshold.to(device))).float(), dim=-1)


class ExplanetModel(nn.Module):
    def __init__(self, train_data, valid_data, backbone=None, box_convert=None, device=DEFAULT_DEVICE):
        super().__init__()
        self.mode = False
        self.device = device
        self.best_d, self.best_c = None, None
        self.ds_t, self.ds_v = train_data, valid_data
        # base_prob = 1./len(train_data.class_list)  # .5 np.nan .5 np.nan base_prob np.nan base_prob np.nan
        self.history = {'bin_acc': [], 'loss_dt': [], 'bin_acc_v': [], 'loss_dv': [],  # TODO - explicability, bbox
                        'cat_acc': [], 'loss_ct': [], 'cat_acc_v': [], 'loss_cv': [], 'shap_ged': []}
        if isinstance(backbone, models.Inception3):
            # Based on: https://github.com/ivanDonadello/Food-Categories-Classification
            self.detector = backbone  # if backbone else resnet50(pretrained=True)
            self.detector.dropout = nn.Identity()           # Moved
            self.detector.fc = nn.Sequential(
                # nn.AdaptiveAvgPool2d(1),                    # Already in inception_v3 output
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Dropout(p=.5),                           # Moved here from inception_v3 output
                nn.Linear(2048, len(self.ds_t.part_list)),
                nn.Sigmoid(),                               # Included in nn.BCEWithLogitsLoss()
            );  self.detector.to(self.device)

            self.optimizer_d = torch.optim.Adam(self.detector.parameters())
            self.scheduler_d = None
        else:  # TODO - take into account which output layer to overwrite (depending on backbone)
            # Based on: https://github.com/JulesSanchez/X-NeSyL
            self.detector = backbone if backbone else models.resnet50(pretrained=True)
            self.detector.fc = nn.Sequential(
                nn.Linear(2048, len(self.ds_t.part_list)),
                nn.Sigmoid(),                               # Included in nn.BCEWithLogitsLoss()
            );  self.detector.to(self.device)

            self.optimizer_d = torch.optim.SGD(self.detector.parameters(), lr=.0003, momentum=.9)
            self.scheduler_d = torch.optim.lr_scheduler.StepLR(self.optimizer_d, 9, gamma=.1, last_epoch=-1)
        if not isinstance(self.detector.fc[-1], nn.Sigmoid):
            self.criterion_d = nn.BCEWithLogitsLoss()
        else:
            self.criterion_d = nn.BCELoss()

        self.detection_threshold = torch.tensor([.5]).to(self.device)
        if box_convert == 'box':
            def box_scores_to_array(x: dict, dataset=train_data):
                """TODO - documentation
                Add scores per object part"""
                agg = [0.]*len(dataset.part_list)
                for i, l in enumerate(x['labels'].detach().cpu().numpy()):
                    agg[l] += x['scores'][i].item()
                return torch.tensor(agg, dtype=torch.float)
            self.detect_convert = box_scores_to_array
        elif box_convert == 'detections':
            self.detect_convert = lambda x: torch.gt(x.detach(), self.detection_threshold).float()
        elif box_convert and box_convert != 'probabilities':
            self.detect_convert = box_convert
        else:  # if box_convert == :  # Probabilities or raw output
            self.detect_convert = lambda x: x.detach().float()
        self.classifier = None

    def forward(self, x: torch.Tensor):
        assert self.classifier is not None
        dtc = self.best_d if self.best_d else self.detector
        cls = self.best_c if self.best_c else self.classifier

        self.train(self.mode)
        return cls(self.detect_convert(dtc(x)))

    def train(self, mode: bool = True):
        self.mode = mode
        self.detector.train(mode)
        if self.classifier:
            self.classifier.train(mode)
        return self

    def train_one_epoch(self, data, valid, epochs_c=25, classifier_neurons=11):
        self.classifier = nn.Sequential(
            nn.Linear(len(data.dataset.part_list), classifier_neurons), nn.ReLU(),
            nn.Linear(classifier_neurons, len(data.dataset.class_list)), nn.Softmax(dim=-1)
        );  self.classifier.to(self.device)  #
        optimizer_c = torch.optim.Adam(self.classifier.parameters())
        criterion_c = ShapBackLoss(data.dataset, data.dataset[:][1][0], self.classifier, self.device)
        # test_c = nn.CrossEntropyLoss()
        # criterion_c = lambda x, y, z: test_c(x, y)

        self.train()
        out_dt, num_t, loss_dt, loss_ct, acc_dt = [], 0, 0., 0., 0.
        for imgs_d, (targets_d, _) in data:
            num_t += len(imgs_d)
            self.optimizer_d.zero_grad()

            output_d = self.detector(imgs_d)
            if isinstance(output_d, models.InceptionOutputs):  # more exceptions
                output_d = output_d.logits
            loss_d = self.criterion_d(output_d, targets_d)  # [0]
            loss_d.backward()
            self.optimizer_d.step()

            acc_dt += torch.sum(binary_accuracy(output_d, targets_d, self.detection_threshold)).item()
            out_dt.append(self.detect_convert(output_d.detach()))
            loss_dt += loss_d.item()
        acc_ct = 0.
        for ep in range(epochs_c):
            loss_ct, acc = 0., 0.
            for bn, (_, (test_d, targets_d)) in enumerate(data):
                optimizer_c.zero_grad()

                output_c = self.classifier(out_dt[bn])  # test_d.float())
                loss_c = criterion_c(output_c, targets_d, out_dt[bn])
                loss_c.backward()
                optimizer_c.step()

                loss_ct += loss_c.item()
                acc += torch.sum(categorical_accuracy(output_c, targets_d)).item()
            if ep >= epochs_c - 1:
                acc_ct = acc
        self.history['bin_acc'].append(acc_dt / num_t)
        self.history['cat_acc'].append(acc_ct / num_t)
        self.history['loss_dt'].append(loss_dt / num_t)
        self.history['loss_ct'].append(loss_ct / num_t)

        self.eval()
        num_v, acc_dv, acc_cv, loss_dv, loss_cv = 0, 0., 0., 0., 0.
        with torch.no_grad():
            for imgs_v, targets_v in valid:
                num_v += len(imgs_v)
                output_dv = self.detector(imgs_v)
                output_cv = self.classifier(output_dv)
                loss_dv += self.criterion_d(output_dv, targets_v[0]).item()
                loss_cv += criterion_c(output_cv, targets_v[1], output_dv).item()
                acc_dv += torch.sum(binary_accuracy(output_dv, targets_v[0], self.detection_threshold)).item()
                acc_cv += torch.sum(categorical_accuracy(output_cv, targets_v[1])).item()
            if self.history['bin_acc_v']:  # TODO - torch.save()
                if self.history['bin_acc_v'][-1] < acc_dv / num_v:
                    self.best_d = self.detector.cpu().state_dict()
                if self.history['cat_acc_v'][-1] < acc_cv / num_v:
                    self.best_c = self.classifier.cpu().state_dict()
        self.history['bin_acc_v'].append(acc_dv / num_v)
        self.history['cat_acc_v'].append(acc_cv / num_v)
        self.history['loss_dv'].append(loss_dv / num_v)
        self.history['loss_cv'].append(loss_cv / num_v)


    def train_one_epoch_bbox(self, data_d, valid_d, data_c, valid_c, epochs_c=25, classifier_neurons=11):
        raise NotImplementedError()  # TODO - bbox version

        self.classifier = nn.Sequential(
            nn.Linear(len(data_d.dataset.part_list), classifier_neurons), nn.ReLU(),
            nn.Linear(classifier_neurons, len(data_d.dataset.class_list)), nn.Softmax(dim=-1)
        );  self.classifier.to(self.device)  #
        optimizer_c = torch.optim.Adam(self.classifier.parameters())
        criterion_c = ShapBackLoss(data_d.dataset, data_d.dataset[:][1][0], self.classifier, self.device)  # TODO
        # test_c = nn.CrossEntropyLoss()
        # criterion_c = lambda x, y, z: test_c(x, y)

        self.train()
        out_dt, num_t, loss_dt, loss_ct, acc_dt = [], 0, 0., 0., 0.
        for imgs_d, targets_d in data_d:
            num_t += len(imgs_d)
            self.optimizer_d.zero_grad()

            output_d = self.detector(imgs_d)
            loss_d = self.criterion_d(output_d, targets_d)
            loss_d.backward()
            self.optimizer_d.step()

            acc_dt += torch.sum(binary_accuracy(output_d, targets_d, self.detection_threshold)).item()
            out_dt.append(self.detect_convert(output_d.detach()))
            loss_dt += loss_d.item()
        acc_ct = 0.
        for ep in range(epochs_c):
            loss_ct, acc = 0., 0.
            for bn, (_, (test_d, targets_d)) in enumerate(data_c):
                optimizer_c.zero_grad()

                output_c = self.classifier(out_dt[bn])
                loss_c = criterion_c(output_c, targets_d, out_dt[bn])
                loss_c.backward()
                optimizer_c.step()

                loss_ct += loss_c.item()
                acc += torch.sum(categorical_accuracy(output_c, targets_d)).item()
            if ep >= epochs_c - 1:
                acc_ct = acc
        self.history['bin_acc'].append(acc_dt / num_t)
        self.history['cat_acc'].append(acc_ct / num_t)
        self.history['loss_dt'].append(loss_dt / num_t)
        self.history['loss_ct'].append(loss_ct / num_t)

        self.eval()
        num_v, acc_dv, acc_cv, loss_dv, loss_cv = 0, 0., 0., 0., 0.
        with torch.no_grad():
            for imgs_v, targets_v in valid_d:  # TODO - properly
                num_v += len(imgs_v)
                output_dv = self.detector(imgs_v)
                loss_dv += self.criterion_d(output_dv, targets_v[0]).item()
                acc_dv += torch.sum(binary_accuracy(output_dv, targets_v[0], self.detection_threshold)).item()
            for imgs_v, targets_v in valid_c:  # TODO - properly
                output_cv = self.classifier(output_dv)
                loss_cv += criterion_c(output_cv, targets_v[1], output_dv).item()
                acc_cv += torch.sum(categorical_accuracy(output_cv, targets_v[1])).item()
            if False:  # self.history['bin_acc_v']:  # TODO - torch.save()
                if self.history['bin_acc_v'][-1] < acc_dv / num_v:
                    self.best_d = self.detector.cpu().state_dict()
                if self.history['cat_acc_v'][-1] < acc_cv / num_v:
                    self.best_c = self.classifier.cpu().state_dict()
        self.history['bin_acc_v'].append(acc_dv / num_v)
        self.history['cat_acc_v'].append(acc_cv / num_v)
        self.history['loss_dv'].append(loss_dv / num_v)
        self.history['loss_cv'].append(loss_cv / num_v)


    def train_model(self, epochs=50, batch_size=8, bboxes=False):  # TODO - bbox version
        # td_data = DataLoader(self.ds_t, batch_size=batch_size, collate_fn=lambda batch: tuple(zip(*batch)))
        # vd_data = DataLoader(self.ds_v, batch_size=batch_size, collate_fn=lambda batch: tuple(zip(*batch)))
        tc_data = DataLoader(self.ds_t.classify, batch_size=batch_size)
        vc_data = DataLoader(self.ds_v.classify, batch_size=batch_size)
        print(f'\rEpoch {0:03d}/{epochs:03d}', end='')
        for ep in range(epochs):
            self.train_one_epoch(tc_data, vc_data)
            if self.scheduler_d:
                self.scheduler_d.step()
            print(f'\nEpoch {ep + 1:03d}/{epochs:03d}'
                  f' (bin_acc = {self.history["bin_acc"][-1]}'
                  f', cat_acc = {self.history["cat_acc"][-1]})', end='')  # TODO - shap_ged
        print('\nFinished')
        return self.history


def visualize_tensor_bbox(img, tens, tars):
    parts = tens['labels'].unique().tolist()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(img.cpu().permute(1, 2, 0))
    for i in range(20):
        xmin, ymin, xmax, ymax = tens['boxes'][i].cpu().detach()
        width, height = xmax - xmin, ymax - ymin

        edgecolor = 'C' + str(parts.index(tens['labels'][i]))
        ax.annotate(f"{tens['labels'][i]}: {tens['scores'][i]:.2f}", xy=(xmax - width + 2, ymin + 10))

        # add bounding boxes to the image
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(rect)
    plt.show()

    lab = tars['labels'].cpu().detach()
    parts = lab.unique().tolist()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(img.cpu().permute(1, 2, 0))
    for i, b in enumerate(tars['boxes']):
        xmin, ymin, xmax, ymax = b.cpu().detach()
        width, height = xmax - xmin, ymax - ymin

        edgecolor = 'C' + str(parts.index(lab[i]))
        ax.annotate(f"{lab[i]}", xy=(xmax - width + 2, ymin + 10))

        # add bounding boxes to the image
        rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')

        ax.add_patch(rect)
    plt.show()


def visualize_img_bbox(ds, idx):
    info = ds.df.loc[[idx], ['img', 'x0', 'y0', 'x1', 'y1', 'part']]
    parts = info.part.unique().tolist()

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    image = plt.imread(info.img.values[0])
    plt.imshow(image)
    for _, row in info.iterrows():
        xmin = row.x0
        if xmin > 0:
            xmax = row.x1
            ymin = row.y0
            ymax = row.y1

            width = xmax - xmin
            height = ymax - ymin

            edgecolor = 'C' + str(parts.index(row.part))
            ax.annotate(row.part, xy=(xmax - width + 2, ymin + height - 2), bbox=dict(boxstyle="round", fc="w"))

            # add bounding boxes to the image
            ax.add_patch(patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none'))
    plt.show()
    print(info.values, idx)


def main():
    fecha = int(time.time())
    os.makedirs(f'./results/{fecha}/code', exist_ok=True)
    root_dir, file_name = os.path.split(os.path.realpath(__file__))
    code, lib, utils = os.path.join(root_dir, file_name), os.path.join(root_dir, "lib"), os.path.join(root_dir, "utils")
    os.system(f'cp -r {code} {lib} {utils} ./results/{fecha}/code')

    # backbones = [mobilenet_v2(pretrained=True), resnet50(pretrained=True)]
    db = 'FFoCat_reduced'  # FFoCat MonuMAI PASCAL
    use_boxes = False
    if 'FFoCat' in db:
        img_size, backbone = 299, models.inception_v3(pretrained=True)
    elif use_boxes:
        img_size, backbone = 224, models.detection.fasterrcnn_resnet50_fpn(True)
    elif db == 'MonuMAI':
        img_size, backbone = 224, models.resnet50(pretrained=True)
    elif db == 'PASCAL':
        img_size, backbone = 224, models.resnet50(pretrained=True)
    else:
        raise Exception('Unknown Dataset')

    data_train, data_valid = get_dataset(db, size=img_size, device=DEFAULT_DEVICE)
    # img_idx = random.Random().randrange(0, data_train.__len__())
    # visualize_img_bbox(data_train, img_idx)

    box_convert = None
    if use_boxes and 'FFoCat' not in db:
        box_convert = 'box'
        in_features = backbone.roi_heads.box_predictor.cls_score.in_features
        backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(data_train.part_list))

    model = ExplanetModel(data_train, data_valid, backbone=backbone, box_convert=box_convert)
    hist = model.train_model(epochs=50)
    print(pd.DataFrame(hist).T)
    for k, v in hist.items():
        plt.plot(v, label=k)
        # print(k, '\n', v)
    plt.legend()
    plt.show()

    torch.save(model, f'./results/{fecha}/{backbone.__name__}.pth')  # TODO - .state_dict()
    # exit(0)


NOTEBOOK = False and torch.cuda.is_available()
if __name__ == '__main__' or NOTEBOOK:
    # TODO - args.parse
    main()

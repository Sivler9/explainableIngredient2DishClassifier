import time
import copy
import math
import random
import os.path
import pickle

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


def binary_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=torch.FloatTensor([.5]), device=DEFAULT_DEVICE):
    return torch.mean(torch.eq(torch.gt(y_true, 0.), torch.gt(y_pred.detach(), threshold.to(device))).float(), dim=-1)


class ExplanetModel(nn.Module):
    def __init__(self, train_data, valid_data, backbone=None, box_convert=None, device=DEFAULT_DEVICE):
        super().__init__()
        self.mode = False
        self.device = device
        self.best_d, self.best_c = None, None
        self.ds_t, self.ds_v = train_data, valid_data
        # base_prob = 1./len(train_data.class_list)  # .5 np.nan .5 np.nan ... base_prob np.nan base_prob np.nan
        self.history = {'bin_acc': [], 'loss_dt': [], 'bin_acc_v': [], 'loss_dv': [], 'shap_ged': [],  # TODO - bbox
                        'cat_acc': [], 'loss_ct': [], 'cat_acc_v': [], 'loss_cv': []}  # , 'shap_ged_v': []
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
            self.criterion_d = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterion_d = nn.BCELoss(reduction='none')

        self.detection_threshold = torch.tensor([.5]).to(self.device)
        if box_convert == 'box':
            def box_scores_to_array(x: dict, dataset=train_data):
                """Aggregate scores per object part
                TODO - docs
                :param x:
                :param dataset:
                :return:
                """
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
        );  self.classifier.to(self.device)
        optimizer_c = torch.optim.Adam(self.classifier.parameters())
        samples = data.dataset[np.random.choice(int(len(data.dataset)), int(.1*len(data.dataset)), False)][1][1]
        criterion_c = ShapBackLoss(data.dataset, samples, self.classifier, self.device)

        self.detector.eval()
        self.classifier.train()
        shap_ged, loss_ct, acc_ct = 0., 0., 0.
        shap_coeffs = []
        for ep in range(epochs_c):
            loss_ct = 0.
            for imgs_t, (targets_t, parts_t, clases_t) in data:
                optimizer_c.zero_grad()

                input_ct = self.detect_convert(self.detector(imgs_t).detach())
                output_ct = self.classifier(input_ct)

                loss_c = criterion_c(output_ct, clases_t)
                loss_c.backward()
                optimizer_c.step()

                if ep >= epochs_c - 1:
                    loss_ct += loss_c.item()*len(targets_t)
                    acc_ct += torch.sum(categorical_accuracy(output_ct, clases_t)).item()
                    sc, ged = criterion_c.shap_coefficient(output_ct, clases_t, input_ct)
                    shap_coeffs.append(sc)
                    shap_ged += sum(ged)

        self.detector.train()
        num_t, loss_dt, acc_dt = 0, 0., 0.
        for bn, (imgs_t, (targets_t, parts_t, clases_t)) in enumerate(data):
            num_t += len(targets_t)
            self.optimizer_d.zero_grad()

            output_d = self.detector(imgs_t)
            if isinstance(output_d, models.InceptionOutputs):  # TODO - more exceptions
                output_d = output_d.logits  # Ignore aux_logits

            crit_d = torch.mean(self.criterion_d(output_d, parts_t), dim=-1)
            loss_d = torch.dot(crit_d, shap_coeffs[bn])/len(crit_d)
            loss_d.backward()
            self.optimizer_d.step()

            acc_dt += torch.sum(binary_accuracy(output_d, parts_t, self.detection_threshold)).item()
            loss_dt += loss_d.item()*len(targets_t)
        self.history['bin_acc'].append(acc_dt/num_t)
        self.history['cat_acc'].append(acc_ct/num_t)
        self.history['loss_dt'].append(loss_dt/num_t)
        self.history['loss_ct'].append(loss_ct/num_t)
        self.history['shap_ged'].append(shap_ged/num_t)

        self.eval()
        num_v, ged_v, acc_dv, acc_cv, loss_dv, loss_cv = 0, 0., 0., 0., 0., 0.
        with torch.no_grad():
            for imgs_v, (targets_v, parts_v, clases_v) in valid:
                num_v += len(targets_v)
                output_dv = self.detector(imgs_v)
                input_cv = self.detect_convert(output_dv)
                output_cv = self.classifier(input_cv)
                crit_dv = torch.mean(self.criterion_d(output_dv, parts_v))  # , dim=-1)
                # sc, ged = criterion_c.shap_coefficient(output_cv, clases_v, input_cv)
                loss_dv += len(targets_v)*crit_dv.item()  # (torch.dot(crit_dv, sc)/len(crit_dv)).item()
                acc_dv += torch.sum(binary_accuracy(output_dv, parts_v, self.detection_threshold)).item()
                # ged_v += sum(ged)
                loss_cv += criterion_c(output_cv, clases_v).item()*len(targets_v)
                acc_cv += torch.sum(categorical_accuracy(output_cv, clases_v)).item()
            if False and self.history['bin_acc_v']:  # TODO - torch.save()
                if self.history['bin_acc_v'][-1] < acc_dv / num_v:
                    self.best_d = self.detector.cpu().state_dict()
                if self.history['cat_acc_v'][-1] < acc_cv / num_v:
                    self.best_c = self.classifier.cpu().state_dict()
        # self.history['shap_ged_v'].append(ged_v/num_v)
        self.history['bin_acc_v'].append(acc_dv/num_v)
        self.history['cat_acc_v'].append(acc_cv/num_v)
        self.history['loss_dv'].append(loss_dv/num_v)
        self.history['loss_cv'].append(loss_cv/num_v)

    def train_model(self, epochs=50, batch_size=16, bboxes=False):  # TODO - bbox version
        t_data = DataLoader(self.ds_t, batch_size=batch_size, collate_fn=shap_collate_fn)
        v_data = DataLoader(self.ds_v, batch_size=batch_size, collate_fn=shap_collate_fn)
        print(f'[INFO] training EXPLANet ...\nEpoch {0:03d}/{epochs:03d}', end='')
        for ep in range(epochs):
            self.train_one_epoch(t_data, v_data)
            if self.scheduler_d:
                self.scheduler_d.step()
            print(f'\nEpoch {ep + 1:03d}/{epochs:03d}'
                  f' (bin_acc = {self.history["bin_acc"][-1]:.5f}|{self.history["bin_acc_v"][-1]:.5f}'
                  f', cat_acc = {self.history["cat_acc"][-1]:.5f}|{self.history["cat_acc_v"][-1]:.5f}'
                  f', shap_ged = {self.history["shap_ged"][-1]:.5f})', end='')
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
        ax.annotate(f'{lab[i]}', xy=(xmax - width + 2, ymin + 10))

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
            ax.annotate(row.part, xy=(xmax - width + 2, ymin + height - 2), bbox=dict(boxstyle='round', fc='w'))

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
    db = 'FFoCat'  # FFoCat _reduced _tiny MonuMAI PASCAL
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

    # TODO - resume
    save_dir = f'./results/{fecha}/'
    torch.save(model.classifier.state_dict(), f'{save_dir}/classify_{backbone.__class__.__name__}.pth')
    torch.save(model.detector.state_dict(), f'{save_dir}/detect_{backbone.__class__.__name__}.pth')
    # torch.save(model, f'./results/{fecha}/{backbone.__class__.__name__}.pth')  # TODO - .state_dict()

    with open(f'{save_dir}/hist_{backbone.__class__.__name__}.pkl', 'wb') as pickle_file:
        pickle.dump(hist, pickle_file)
    # df_hist = pd.DataFrame(hist)
    # print(df_hist.T)

    plt.figure()
    for k, v in hist.items():
        if 'loss' in k:
            plt.plot(v, label=k)  # print(k, '\n', v)
    plt.legend()
    plt.show()
    plt.savefig(f'{save_dir}/0.pdf')

    plt.figure()
    for k, v in hist.items():
        if 'acc' in k:
            plt.plot(v, label=k)
    plt.legend()
    plt.show()
    plt.savefig(f'{save_dir}/1.pdf')

    plt.figure()
    for k, v in hist.items():
        if 'ged' in k:
            plt.plot(v, label=k)
    plt.legend()
    plt.show()
    plt.savefig(f'{save_dir}/2.pdf')

    # exit(0)


NOTEBOOK = False and torch.cuda.is_available()
if __name__ == '__main__' or NOTEBOOK:
    # TODO - args.parse
    main()

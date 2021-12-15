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

SAVE_DIR = f'./results/{int(time.time())}/'
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
            def box_scores_to_array(x: list[dict], dataset=train_data):
                """Aggregate scores per object part
                TODO - docs
                :param x:
                :param dataset:
                :return:
                """
                aggs = []
                for d in x:
                    agg = [0.]*len(dataset.part_list)
                    for i, l in enumerate(d['labels'].detach().cpu().numpy()):
                        agg[l] += d['scores'][i].item()
                    aggs.append(torch.tensor(agg, dtype=torch.float))
                return torch.stack(aggs).to(self.device)
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

    def save_checkpoint(self, save_dir=SAVE_DIR, name='check'):
        # TODO - optimizer, etc ...
        torch.save(self.detector.state_dict(), f'{save_dir}/detect_{name}.pth')
        torch.save(self.classifier.state_dict(), f'{save_dir}/classify_{name}.pth')

    def load_checkpoint(self, path_detector='', path_classifier=''):
        # TODO - optimizer, etc ...
        if path_detector:
            self.detector.load_state_dict(torch.load(path_detector))
        if path_classifier:
            self.classifier.load_state_dict(torch.load(path_classifier))

    def train_one_epoch(self, data, valid, epochs_c=25, classifier_neurons=11):
        self.classifier = nn.Sequential(
            nn.Linear(len(data.dataset.part_list), classifier_neurons), nn.ReLU(),
            # nn.Linear(classifier_neurons, classifier_neurons), nn.ReLU(),
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
        # self.history['shap_ged_v'].append(ged_v/num_v)
        self.history['bin_acc_v'].append(acc_dv/num_v)
        self.history['cat_acc_v'].append(acc_cv/num_v)
        self.history['loss_dv'].append(loss_dv/num_v)
        self.history['loss_cv'].append(loss_cv/num_v)

    def train_one_epoch_box(self, data, valid, epochs_c=25, classifier_neurons=11):
        self.classifier = nn.Sequential(
            nn.Linear(len(data.dataset.part_list), classifier_neurons), nn.ReLU(),
            # nn.Linear(classifier_neurons, classifier_neurons), nn.ReLU(),
            nn.Linear(classifier_neurons, len(data.dataset.class_list)), nn.Softmax(dim=-1)
        );  self.classifier.to(self.device)
        optimizer_c = torch.optim.Adam(self.classifier.parameters())
        samples = data.dataset[np.random.choice(int(len(data.dataset)), min(int(.3*len(data.dataset)), 200), False)][1][1]
        criterion_c = nn.CrossEntropyLoss()

        self.detector.eval()
        self.classifier.train()
        shap_ged, loss_ct, acc_ct = 0., 0., 0.
        shap_coeffs = []
        for ep in range(epochs_c):
            loss_ct = 0.
            for imgs_t, (targets_t, parts_t, clases_t) in data:
                optimizer_c.zero_grad()

                input_ct = self.detect_convert(self.detector([imgs_t]))
                output_ct = self.classifier(input_ct)

                loss_c = criterion_c(output_ct, clases_t.view(-1))
                loss_c.backward()
                optimizer_c.step()

                if ep >= epochs_c - 1:
                    loss_ct += loss_c.item()*len(targets_t)
                    acc_ct += torch.sum(categorical_accuracy(output_ct, clases_t)).item()
                    sc, ged = criterion_c.shap_coefficient(output_ct, clases_t, input_ct)
                    shap_coeffs.append(sc)
                    shap_ged += sum(ged)
            if ep == epochs_c - 2:
                criterion_c = ShapBackLoss(data.dataset, samples, self.classifier, self.device)

        num_t, loss_dt, acc_dt = 0, 0., 0.
        for bn, (imgs_t, (targets_t, parts_t, clases_t)) in enumerate(data):
            num_t += len(targets_t)
            self.optimizer_d.zero_grad()
            self.detector.train()
            loss_d = self.detector([imgs_t], targets_t)
            loss_d = sum(l for l in loss_d.values())*shap_coeffs[bn]

            self.detector.eval()
            output_dt = self.detect_convert(self.detector([imgs_t]))
            acc_dt += torch.sum(binary_accuracy(output_dt, parts_t, self.detection_threshold)).item()

            loss_d.backward()
            self.optimizer_d.step()
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
                self.detector.eval()
                output_dv = self.detector([imgs_v])
                input_cv = self.detect_convert(output_dv)
                output_cv = self.classifier(input_cv)
                self.detector.train()
                loss_d = self.detector([imgs_v])
                # sc, ged = criterion_c.shap_coefficient(output_cv, clases_v, input_cv)
                loss_dv += len(targets_v)*sum(l for l in loss_d.values()).item()
                acc_dv += torch.sum(binary_accuracy(output_dv, parts_v, self.detection_threshold)).item()
                # ged_v += sum(ged)
                loss_cv += criterion_c(output_cv, clases_v).item()*len(targets_v)
                acc_cv += torch.sum(categorical_accuracy(output_cv, clases_v)).item()
        # self.history['shap_ged_v'].append(ged_v/num_v)
        self.history['bin_acc_v'].append(acc_dv/num_v)
        self.history['cat_acc_v'].append(acc_cv/num_v)
        self.history['loss_dv'].append(loss_dv/num_v)
        self.history['loss_cv'].append(loss_cv/num_v)

    def train_model(self, epochs=50, batch_size=16, patience=5, delta=1e-3, bboxes=False):
        if bboxes:
            batch_size = 1
        t_data = DataLoader(self.ds_t, batch_size=batch_size, collate_fn=shap_collate_fn)
        v_data = DataLoader(self.ds_v, batch_size=batch_size, collate_fn=shap_collate_fn)
        print(f'[DATE] {time.strftime("%Y-%b-%d %H:%M:%S", time.localtime())}')
        print(f'[INFO] training EXPLANet ...\nEpoch {0:03d}/{epochs:03d}', end='')

        best_score, counter = np.inf, 0  # early_stopping
        for ep in range(epochs):
            if bboxes:
                self.train_one_epoch_box(t_data, v_data)
            else:
                self.train_one_epoch(t_data, v_data)
            if self.scheduler_d:
                self.scheduler_d.step()
            print(f'\nEpoch {ep + 1:03d}/{epochs:03d}'
                  f' (bin_acc = {self.history["bin_acc"][-1]:.5f}|{self.history["bin_acc_v"][-1]:.5f}'
                  f', cat_acc = {self.history["cat_acc"][-1]:.5f}|{self.history["cat_acc_v"][-1]:.5f}'
                  f', shap_ged = {self.history["shap_ged"][-1]:.5f})', end='')
            score = self.history['loss_dv'][-1]
            if score > best_score + delta:
                counter += 1
                if counter >= patience:
                    break  # early stop
            else:
                self.save_checkpoint(name='tmp')
                best_score = score
                counter = 0
        self.load_checkpoint(f'{SAVE_DIR}/detect_tmp.pth', f'{SAVE_DIR}/classify_tmp.pth')
        print(f'\n[DATE] {time.strftime("%Y-%b-%d %H:%M:%S", time.localtime())}')
        return self.history


def visualize_tensor_bbox(img, tens, tars):
    raise DeprecationWarning()
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
    raise DeprecationWarning()
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
    # Reproducibility
    random.seed(621)
    np.random.seed(621)
    torch.manual_seed(621)

    os.makedirs(f'{SAVE_DIR}/code', exist_ok=True)
    root_dir, file_name = os.path.split(os.path.realpath(__file__))
    code, lib, utils = os.path.join(root_dir, file_name), os.path.join(root_dir, 'lib'), os.path.join(root_dir, 'utils')
    os.system(f'cp -r {code} {lib} {utils} {SAVE_DIR}/code')

    for db in ['FFoCat_tiny', 'MonuMAI', 'PASCAL', 'FFoCat'][:]:  # FFoCat _reduced _tiny MonuMAI PASCAL
        for backbone in [models.inception_v3(pretrained=True), models.detection.fasterrcnn_resnet50_fpn(True),
                         models.resnet50(pretrained=True), models.mobilenet_v2(pretrained=True)][1::-1]:
            use_boxes = isinstance(backbone, models.detection.FasterRCNN)
            if use_boxes and 'FFoCat' in db:
                continue

            img_size = 299 if isinstance(backbone, models.Inception3) else 224
            data_train, data_valid = get_dataset(db, size=img_size, device=DEFAULT_DEVICE)
            if use_boxes:
                box_convert = 'box'
                in_features = backbone.roi_heads.box_predictor.cls_score.in_features
                backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(data_train.part_list))
            else:
                box_convert = 'detections'  # TODO - try others
            name_id = f'{db}_{backbone.__class__.__name__}{"_box" * use_boxes}'
            print(f'[INFO] {db} {backbone.__class__.__name__}')

            model = ExplanetModel(data_train, data_valid, backbone=backbone, box_convert=box_convert)
            hist = model.train_model(epochs=5 if 'tiny' in db else 50, batch_size=16, bboxes=use_boxes)
            model.save_checkpoint(SAVE_DIR, name_id)

            with open(f'{SAVE_DIR}/hist_{name_id}.pkl', 'wb') as pickle_file:
                pickle.dump(hist, pickle_file)
            # df_hist = pd.DataFrame(hist)
            # print(df_hist.T)

            for n, topic in enumerate(['loss', 'acc', 'ged']):
                plt.figure()
                for k, v in hist.items():
                    if topic in k:
                        plt.plot(v, label=k)  # print(k, '\n', v)
                plt.legend()
                plt.savefig(f'{SAVE_DIR}/{n}_{name_id}.pdf')
                plt.show()
    # exit(0)


NOTEBOOK = False and torch.cuda.is_available()
if __name__ == '__main__' or NOTEBOOK:
    # TODO - args.parse
    main()

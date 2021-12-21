import time
import pickle
import random
import os.path

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


def move_to_device(data, device=DEFAULT_DEVICE):
    imgs, (targets, parts, clases) = data
    imgs = imgs.to(device)
    parts = parts.to(device)
    clases = clases.to(device)
    for t in targets:
        for k in t.keys():
            v = t[k]
            if torch.is_tensor(v):
                t[k] = t[k].to(device)
    return imgs, (targets, parts, clases)


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

        def replace_last_layer(model, out_feat):  # TODO - Detect if the same model is passes twice+
            if isinstance(model, models.detection.FasterRCNN):
                model.fc = nn.Sequential(
                    nn.Linear(2048, out_feat),
                    nn.Sigmoid(),  # Included in nn.BCEWithLogitsLoss()
                )
                model = model.fc
            else:
                last_candidates = model._modules
                if len(last_candidates):
                    layer = replace_last_layer(next(reversed(last_candidates.values())), out_feat)
                    if not len(layer._modules.keys()):
                        replacement = nn.Sequential(
                            nn.Linear(layer.in_features, out_feat),
                            nn.Sigmoid(),  # Included in nn.BCEWithLogitsLoss()
                        )
                        if isinstance(model, nn.Sequential):
                            model[-1] = replacement
                        else:
                            name = next(reversed(last_candidates.keys()))
                            setattr(model, name, replacement)
                        model = replacement
                    else:
                        model = layer
            return model

        if isinstance(backbone, models.Inception3):
            # Based on: https://github.com/ivanDonadello/Food-Categories-Classification
            self.detector = backbone
            self.detector.dropout = nn.Identity()           # Moved - to match tf2/keras
            self.detector.fc = nn.Sequential(
                # nn.AdaptiveAvgPool2d(1),                    # Already in inception_v3 output
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Dropout(p=.5),                           # Moved - here from inception_v3 output
                nn.Linear(2048, len(self.ds_t.part_list)),
                nn.Sigmoid(),                               # Included in nn.BCEWithLogitsLoss()
            );  self.detector.to(self.device)

            self.optimizer_d = torch.optim.Adam(self.detector.parameters())  # self.detector.fc
            self.scheduler_d = None
            last_layer = self.detector.fc[-1]
        else:
            # Based on: https://github.com/JulesSanchez/X-NeSyL
            self.detector = backbone if backbone else models.resnet50(pretrained=True)
            last_layer = replace_last_layer(self.detector, len(self.ds_t.part_list))
            self.detector.to(self.device)

            self.optimizer_d = torch.optim.SGD(self.detector.parameters(), lr=.0003, momentum=.9)  # last_layer
            self.scheduler_d = torch.optim.lr_scheduler.StepLR(self.optimizer_d, 9, gamma=.1, last_epoch=-1)
            last_layer = last_layer[-1]
        if isinstance(last_layer, nn.Sigmoid):
            self.criterion_d = nn.BCELoss(reduction='none')
        else:
            self.criterion_d = nn.BCEWithLogitsLoss(reduction='none')
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
            nn.Linear(classifier_neurons, classifier_neurons), nn.ReLU(),
            nn.Linear(classifier_neurons, len(data.dataset.class_list)),  # nn.Softmax(dim=-1)
        );  self.classifier.to(self.device)
        optimizer_c = torch.optim.Adam(self.classifier.parameters())
        criterion_c = nn.CrossEntropyLoss()

        self.detector.eval()
        self.classifier.train()
        shap_coeffs = []
        start_f = time.time()
        shap_ged, loss_ct, acc_ct = 0., 0., 0.
        for ep in range(epochs_c):
            start_e = time.time()
            for data_t in data:
                end_f = time.time() - start_f
                imgs_t, (targets_t, parts_t, clases_t) = move_to_device(data_t, device=self.device)
                start_b = time.time()

                optimizer_c.zero_grad()
                with torch.no_grad():
                    input_ct = self.detect_convert(self.detector(imgs_t))
                # input_ct = parts_t
                output_ct = self.classifier(input_ct)

                loss_c = criterion_c(output_ct, clases_t)
                loss_c.backward()
                optimizer_c.step()

                print(f'\rc{ep} [{time.time() - start_e:.1f}s] b[{time.time() - start_b:.3f}s] l[{end_f:.3f}s]'
                      f' (running_loss={loss_c.item():.3f})', end='')
                start_f = time.time()

        num_samples = min(max(5, int(.3 * len(data.dataset))), 300)
        samples = data.dataset[np.random.choice(len(data.dataset), num_samples, False)][1][1]
        criterion_c = ShapBackLoss(data.dataset, samples, self.classifier, self.device)
        num_t, loss_ct = 0, 0.
        start_e = time.time()
        for data_t in data:
            end_f = time.time() - start_f
            imgs_t, (targets_t, parts_t, clases_t) = move_to_device(data_t, device=self.device)
            num_t += len(targets_t)
            start_b = time.time()

            optimizer_c.zero_grad()
            with torch.no_grad():
                input_ct = self.detect_convert(self.detector(imgs_t))
            # input_ct = parts_t
            output_ct = self.classifier(input_ct)

            loss_c = criterion_c(output_ct, clases_t)
            loss_c.backward()
            optimizer_c.step()

            print(f'\rshap [{time.time() - start_e:.1f}s] n{num_t}/{len(data.dataset)} '
                  f'b[{time.time() - start_b:.3f}s] l[{end_f:.3f}s] (running_loss={loss_c.item():.3f})', end='')
            start_f = time.time()

            loss_ct += loss_c.item()*len(targets_t)
            acc_ct += torch.sum(categorical_accuracy(output_ct, clases_t)).item()
            sc, ged = criterion_c.shap_coefficient(output_ct, clases_t, input_ct)
            shap_coeffs.append(sc)
            shap_ged += sum(ged)
        self.history['shap_ged'].append(shap_ged/num_t)
        self.history['loss_ct'].append(loss_ct/num_t)
        self.history['cat_acc'].append(acc_ct/num_t)
        print()

        self.detector.train()
        start_e = time.time()
        loss_dt, acc_dt = 0., 0.
        for bn, data_t in enumerate(data):
            imgs_t, (targets_t, parts_t, clases_t) = move_to_device(data_t, device=self.device)
            self.optimizer_d.zero_grad()
            output_d = self.detector(imgs_t)
            if isinstance(output_d, models.InceptionOutputs):
                output_d = output_d.logits  # Ignore aux_logits

            crit_d = torch.mean(self.criterion_d(output_d, parts_t), dim=-1)
            loss_d = torch.dot(crit_d, shap_coeffs[bn])/len(crit_d)
            loss_d.backward()
            self.optimizer_d.step()

            acc_dt += torch.sum(binary_accuracy(output_d, parts_t, self.detection_threshold)).item()
            loss_dt += loss_d.item()*len(shap_coeffs[0])
            nt = (1 + bn)*len(shap_coeffs[0])
            print(f'\rd [{time.time() - start_e:.3f}] n{nt}/{num_t} '
                  f'(running_loss={loss_dt/nt:.3f}, running_accuracy={acc_dt/nt:.3f})', end='')
        self.history['loss_dt'].append(loss_dt/num_t)
        self.history['bin_acc'].append(acc_dt/num_t)
        print()

        self.eval()
        num_v, ged_v, acc_dv, acc_cv, loss_dv, loss_cv = 0, 0., 0., 0., 0., 0.
        with torch.no_grad():
            for data_v in valid:
                imgs_v, (targets_v, parts_v, clases_v) = move_to_device(data_v, device=self.device)
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
        # TODO - Use RMSE instead of categorical_accuracy
        raise DeprecationWarning  # TODO - Redo
        data.dataset.count = True
        valid.dataset.count = True
        self.classifier = nn.Sequential(
            nn.Linear(len(data.dataset.part_list), classifier_neurons), nn.ReLU(),
            nn.Linear(classifier_neurons, classifier_neurons), nn.ReLU(),
            nn.Linear(classifier_neurons, len(data.dataset.class_list)),  # nn.Softmax(dim=-1)
        );  self.classifier.to(self.device)
        optimizer_c = torch.optim.Adam(self.classifier.parameters())
        criterion_c = nn.CrossEntropyLoss()

        self.detector.eval()
        self.classifier.train()
        shap_coeffs = []
        start_f = time.time()
        shap_ged, loss_ct, acc_ct = 0., 0., 0.
        for ep in range(epochs_c):
            start_e = time.time()
            for data_t in data:
                end_f = time.time() - start_f
                imgs_t, (targets_t, parts_t, clases_t) = move_to_device(data_t, device=self.device)
                start_b = time.time()

                optimizer_c.zero_grad()
                with torch.no_grad():
                    input_ct = self.detect_convert(self.detector(imgs_t))
                # input_ct = parts_t
                output_ct = self.classifier(input_ct)

                loss_c = criterion_c(output_ct, clases_t)
                loss_c.backward()
                optimizer_c.step()

                print(f'\rc{ep} [{time.time() - start_e:.1f}s] b[{time.time() - start_b:.3f}s] l[{end_f:.3f}s]'
                      f' (running_loss={loss_c.item():.3f})', end='')
                start_f = time.time()

        num_samples = min(max(5, int(.3 * len(data.dataset))), 300)
        samples = data.dataset[np.random.choice(len(data.dataset), num_samples, False)][1][1]
        criterion_c = ShapBackLoss(data.dataset, samples, self.classifier, self.device)
        num_t, loss_ct = 0, 0.
        start_e = time.time()
        for data_t in data:
            end_f = time.time() - start_f
            imgs_t, (targets_t, parts_t, clases_t) = move_to_device(data_t, device=self.device)
            num_t += len(targets_t)
            start_b = time.time()

            optimizer_c.zero_grad()
            with torch.no_grad():
                input_ct = self.detect_convert(self.detector(imgs_t))
            # input_ct = parts_t
            output_ct = self.classifier(input_ct)

            loss_c = criterion_c(output_ct, clases_t)
            loss_c.backward()
            optimizer_c.step()

            print(f'\rshap [{time.time() - start_e:.1f}s] n{num_t}/{len(data.dataset)} '
                  f'b[{time.time() - start_b:.3f}s] l[{end_f:.3f}s] (running_loss={loss_c.item():.3f})', end='')
            start_f = time.time()

            loss_ct += loss_c.item()*len(targets_t)
            acc_ct += torch.sum(categorical_accuracy(output_ct, clases_t)).item()
            sc, ged = criterion_c.shap_coefficient(output_ct, clases_t, input_ct)
            shap_coeffs.append(sc)
            shap_ged += sum(ged)
        self.history['shap_ged'].append(shap_ged/num_t)
        self.history['loss_ct'].append(loss_ct/num_t)
        self.history['cat_acc'].append(acc_ct/num_t)
        print()

        self.detector.train()
        start_e = time.time()
        loss_dt, acc_dt = 0., 0.
        for bn, data_t in enumerate(data):
            imgs_t, (targets_t, parts_t, clases_t) = move_to_device(data_t, device=self.device)
            self.optimizer_d.zero_grad()
            output_d = self.detector(imgs_t)
            if isinstance(output_d, models.InceptionOutputs):
                output_d = output_d.logits  # Ignore aux_logits

            crit_d = torch.mean(self.criterion_d(output_d, parts_t), dim=-1)
            loss_d = torch.dot(crit_d, shap_coeffs[bn])/len(crit_d)
            loss_d.backward()
            self.optimizer_d.step()

            acc_dt += torch.sum(binary_accuracy(output_d, parts_t, self.detection_threshold)).item()
            loss_dt += loss_d.item()*len(shap_coeffs[0])
            nt = (1 + bn)*len(shap_coeffs[0])
            print(f'\rd [{time.time() - start_e:.3f}] n{nt}/{num_t} '
                  f'(running_loss={loss_dt/nt:.3f}, running_accuracy={acc_dt/nt:.3f})', end='')
        self.history['loss_dt'].append(loss_dt/num_t)
        self.history['bin_acc'].append(acc_dt/num_t)
        print()

        self.eval()
        num_v, ged_v, acc_dv, acc_cv, loss_dv, loss_cv = 0, 0., 0., 0., 0., 0.
        with torch.no_grad():
            for data_v in valid:
                imgs_v, (targets_v, parts_v, clases_v) = move_to_device(data_v, device=self.device)
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

    def train_model(self, epochs=50, batch_size=16, patience=5, delta=1e-3, bboxes=False):
        if bboxes:
            batch_size = 1
        t_data = DataLoader(self.ds_t, batch_size=batch_size, collate_fn=shap_collate_fn, num_workers=8)
        v_data = DataLoader(self.ds_v, batch_size=batch_size, collate_fn=shap_collate_fn, num_workers=8)
        print(f'[DATE] {time.strftime("%Y-%b-%d %H:%M:%S", time.localtime())}\n'
              f'[INFO] training EXPLANet ...\n\nEpoch {0:03d}/{epochs:03d}', end='\n')

        start_train = time.time()
        neurons = len(self.ds_t.class_list)*3 - 1
        best_score, counter = np.inf, 0  # early_stopping
        for ep in range(epochs):
            start_time = time.time()
            if bboxes:
                self.train_one_epoch_box(t_data, v_data, classifier_neurons=neurons)
            else:
                self.train_one_epoch(t_data, v_data, classifier_neurons=neurons)
            if self.scheduler_d:
                self.scheduler_d.step()
            score = self.history['loss_dv'][-1]
            if score > best_score + delta:
                counter += 1
            else:
                self.save_checkpoint(name='tmp')
                best_score = score
                counter = 0
            print(f'\nEpoch {ep + 1:03d}/{epochs:03d}'
                  f' (bin_acc={self.history["bin_acc"][-1]:.5f}|{self.history["bin_acc_v"][-1]:.5f}'
                  f', cat_acc={self.history["cat_acc"][-1]:.5f}|{self.history["cat_acc_v"][-1]:.5f}'
                  f', loss_dv={self.history["loss_dv"][-1]:.5f}, shap_ged={self.history["shap_ged"][-1]:.5f})'
                  f' [patience: {patience - counter}/{patience}, time: {time.time() - start_time:.5f}s]', end='\n')
            if counter >= patience:
                break  # early stop
        self.load_checkpoint(f'{SAVE_DIR}/detect_tmp.pth', f'{SAVE_DIR}/classify_tmp.pth')
        print(f'[INFO] training finished - {time.time() - start_train}s\n'
              f'[DATE] {time.strftime("%Y-%b-%d %H:%M:%S", time.localtime())}')
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

    # Set up
    os.makedirs(f'{SAVE_DIR}/code', exist_ok=True)
    root_dir, file_name = os.path.split(os.path.realpath(__file__))
    code, lib, utils = os.path.join(root_dir, file_name), os.path.join(root_dir, 'lib'), os.path.join(root_dir, 'utils')
    os.system(f'cp -r {code} {lib} {utils} {SAVE_DIR}/code')

    # Execution
    for backbone in [models.inception_v3, models.resnet50, models.mobilenet_v2,
                     models.detection.fasterrcnn_resnet50_fpn]:
        for db in ['MonuMAI', 'PASCAL', 'FFoCat'][::]:  # FFoCat _reduced _tiny MonuMAI PASCAL
            backbone = backbone(pretrained=True)
            use_boxes = isinstance(backbone, models.detection.FasterRCNN)
            if use_boxes and 'FFoCat' in db:
                continue

            img_size = 299 if isinstance(backbone, models.Inception3) else 224
            data_train, data_valid = get_dataset(db, size=img_size, device=torch.device('cpu'))
            if use_boxes:
                box_convert = 'box'
                in_features = backbone.roi_heads.box_predictor.cls_score.in_features
                backbone.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(data_train.part_list))
            else:
                box_convert = 'detections'  # TODO - try others
            name_id = f'{db}_{backbone.__class__.__name__}{"_box" * use_boxes}'
            print(f'[INFO] now running {db} {backbone.__class__.__name__}')

            model = ExplanetModel(data_train, data_valid, backbone=backbone, box_convert=box_convert)
            hist = model.train_model(epochs=3 if '_t' in db else 50, batch_size=64, bboxes=use_boxes)
            model.save_checkpoint(SAVE_DIR, name_id)

            with open(f'{SAVE_DIR}/hist_{name_id}.pkl', 'wb') as pickle_file:
                pickle.dump(hist, pickle_file)
            # df_hist = pd.DataFrame(hist)
            # print(df_hist.T)

            for n, topic in enumerate(['loss', 'acc', 'ged']):
                plt.figure()
                for k, v in hist.items():
                    if topic in k:
                        plt.plot(v, label=k)
                        # print(k, '\n', v)
                plt.legend()
                plt.savefig(f'{SAVE_DIR}/{n}_{name_id}.pdf')
                plt.show()

    # Clean up
    os.system(f'rm {SAVE_DIR}/*_tmp.pth')


NOTEBOOK = False and torch.cuda.is_available()
if __name__ == '__main__' or NOTEBOOK:
    import warnings
    warnings.filterwarnings('error')
    # torch.multiprocessing.set_start_method('spawn')
    # TODO - args.parse
    main()

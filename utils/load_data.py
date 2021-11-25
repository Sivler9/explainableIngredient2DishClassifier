"""
TODO - docs
"""
import os

import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as Et
import albumentations as t  # from torchvision import transforms as t

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

from sklearn.preprocessing import MultiLabelBinarizer

NOTEBOOK = False  # ('Data/FFoCat_reduced' if NOTEBOOK else 'Data/FFoCat_tiny')


def _build_food_labels_dict(root_dir='Data/FFoCat', csv_loc=None) -> dict:
    csv_loc = csv_loc if csv_loc else os.path.join(root_dir, 'food_food_category_map.tsv')
    recipe_food_map = np.genfromtxt(csv_loc, delimiter="\t", dtype=str)

    recipe_food_dict = {}
    for recipe_food in recipe_food_map:
        recipe = recipe_food[0] + '_' + recipe_food[1]
        if recipe not in recipe_food_dict:
            recipe_food_dict[recipe] = []
        recipe_food_dict[recipe].append(recipe_food[2])
    return recipe_food_dict


def _get_food_data(cmap, root_dir='Data/FFoCat', train=False) -> pd.DataFrame:
    mode = 'train' if train else 'valid'
    data = {'img': [], 'class': [], 'boxes': []}
    for r, d, f in os.walk(os.path.join(root_dir, mode)):
        for dd in d:
            boxs = [[-1, -1, -1, -1, label] for label in cmap[dd]]
            for rr, _, ff in os.walk(os.path.join(r, dd)):
                for fff in ff:
                    data['img'].append(os.path.join(rr, fff))
                    data['boxes'].append(boxs)
                    data['class'].append(dd)
    df = pd.DataFrame(data)
    df = df.explode('boxes')
    df['is_part'] = True
    df[['x0', 'y0', 'x1', 'y1', 'part']] = df.boxes.tolist()
    return df  # .drop('boxes')


def _get_pascal_data(cmap, root_dir='Data/LTN_ACM_SAC17/pascalpart_dataset/JPEGImages',
                     train=False) -> pd.DataFrame:
    part_list = set([prt for cls in cmap.values() for prt in cls])
    mode = '_train.txt' if train else '_test.txt'  # TODO - val.csv
    data = {'img': [], 'class': [], 'boxes': []}
    for cls in cmap.keys():
        df_cls = pd.read_csv(os.path.join(root_dir, '..', 'ImageSets', 'Main', cls + mode),
                             header=None, delim_whitespace=True)
        for c in df_cls[df_cls[1] == 1][0]:

            img = os.path.join(root_dir, f'{c:06}.jpg')
            if img in data['img']:
                continue
            boxs, groups = [], set()
            for n in Et.parse(os.path.join(root_dir, '..', 'Annotations', f'{c:06}.xml')).findall('object'):
                box = [int(n.find('bndbox').find(bnd).text) for bnd in ['xmin', 'ymin', 'xmax', 'ymax']]
                prt = n.find('name').text
                if prt in part_list:
                    boxs.append([box[0], box[1], box[2] + (box[0] == box[2]),
                                 box[3] + (box[1] == box[3]), prt, prt in part_list])
                if prt in cmap:
                    groups.add(prt)
            if len(groups) < 2 and len(boxs):  # Imgs w/only 1 global/macro label
                data['boxes'].append(boxs)
                data['class'].append(cls)
                data['img'].append(img)
    df = pd.DataFrame(data)
    df = df.explode('boxes')
    df[['x0', 'y0', 'x1', 'y1', 'part', 'is_part']] = df.boxes.tolist()
    return df  # .drop('boxes')


def _get_monumai_data(cmap, root_dir='Data/OD-MonuMAI/MonuMAI_dataset/',
                      train=False) -> pd.DataFrame:
    mode = 'train.csv' if train else 'test.csv'  # TODO - val.csv
    data = {'img': [], 'class': [], 'boxes': []}
    for folder in pd.read_csv(os.path.join(root_dir, mode)).path:
        d = folder.split('dataset')[-1][1:]
        *_, cls, fn = d.split(os.path.sep)
        data['img'].append(os.path.join(root_dir, d))
        data['class'].append(cls)
        boxs = []
        for n in Et.parse(os.path.join(root_dir, cls, 'xml', fn.split('.')[0] + '.xml')).getroot().findall('object'):
            box = [int(n.find('bndbox').find(c).text) for c in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxs.append([*box, n.find('name').text])
        data['boxes'].append(boxs)
    df = pd.DataFrame(data)
    df = df.explode('boxes')
    df['is_part'] = True
    df[['x0', 'y0', 'x1', 'y1', 'part']] = df.boxes.tolist()
    return df  # .drop('boxes')


class ShapImageDataset(Dataset):
    class ClassifyView(Dataset):
        def __init__(self, ds):
            self.ds, self.class_list, self.part_list = ds, ds.class_list, ds.part_list
            self.df, self.tf, self.mlb, self.cmap, self.device = ds.df, ds.tf, ds.mlb, ds.cmap, ds.device

        def __len__(self):
            return self.ds.__len__()

        def __getitem__(self, item):
            self.ds._classify_mode = True
            return self.ds[item]

    def __init__(self, dataframe: pd.DataFrame, class_map: dict, transforms: t.Compose = ToTensor(),
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.df, self.tf, self.cmap = dataframe[dataframe.is_part], transforms, class_map
        self.part_list = sorted(set(fea for ele in self.cmap.values() for fea in ele))
        self.mlb = MultiLabelBinarizer(classes=self.part_list)
        self.class_list = sorted(self.cmap.keys())
        self._classify_mode = False
        self.device = device

    def __len__(self):
        return self.df.img.unique().__len__()

    @property
    def classify(self):
        return self.ClassifyView(self)

    @property
    def detect(self):
        return self

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, slice):
            step = idx.step if idx.step else 1
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else (self.__len__() - 1)
            idx = range(start, stop, step)

        image_f, boxes, parts, clases = [], [], [], []
        for i in idx:
            img = self.df.loc[[i], ['img']].values[0, 0]
            part = self.df.loc[[i], 'part'].values.tolist()
            clas = self.df.loc[[i], ['class']].values[0, 0]
            box = self.df.loc[[i], ['x0', 'y0', 'x1', 'y1']].values
            image_f.append((img, np.array(Image.open(img).convert('RGB'))))
            clases.append(clas)
            parts.append(part)
            boxes.append(box)

        tmp = self.tf.processors, self.tf.is_check_args
        if self.df.x0.iloc[0] < 0.:
            self.tf.is_check_args = False
            self.tf.processors = {}
        if self._classify_mode:
            augs = [self.tf(image=image_f[i][1], bboxes=boxes[i], parts=parts[i]) for i in range(len(image_f))]

            classes = torch.LongTensor([self.class_list.index(clases[i]) for i in range(len(augs))])
            parts = torch.LongTensor(self.mlb.fit_transform([[p for p in a['parts']] for a in augs]))
            image = torch.stack([a['image'] for a in augs])

            # classes = F.one_hot(classes, num_classes=len(self.classes))
            target = [torch.squeeze(parts.view(-1, len(self.part_list))).float().to(self.device),
                      torch.squeeze(classes.view(-1, 1)).long().to(self.device)]
        elif len(idx) == 1:
            (image_f, image), boxes, parts, clas = image_f[0], boxes[0], parts[0], clases[0]

            aug = self.tf(image=image, bboxes=boxes, parts=parts)
            image, boxes, parts = aug['image'], aug['bboxes'], aug.get('parts', None)
            boxes = torch.FloatTensor(np.array(boxes)[:, :4].astype(np.float32))
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target = {
                'labels': torch.LongTensor([self.part_list.index(p) for p in parts], device=self.device),
                'class': torch.LongTensor([self.class_list.index(clas)], device=self.device),
                # 'iscrowd': torch.zeros((len(parts),), dtype=torch.int64, device=self.device), 'masks': None,
                'area': area.to(self.device), 'boxes': boxes.to(self.device),
                'image_id': torch.LongTensor([idx], device=self.device),
                'file_name': image_f[:1],
            }
        else:
            raise NotImplementedError('Batch support not implemented for the detection format')
        if self.df.x0.iloc[0] < 0.:
            self.tf.processors, self.tf.is_check_args = tmp
        self._classify_mode = False
        return torch.squeeze(image).to(self.device), target


def get_dataset(name='FFoCat', size=224, device=torch.device('cpu')) -> [ShapImageDataset, ShapImageDataset]:
    def get_mean_std(datas):
        means, stds = [], []
        for img in datas:
            stds.append(torch.std(img))
            means.append(torch.mean(img))
        return torch.mean(torch.tensor(means)), torch.mean(torch.tensor(stds))

    out_size = (256*size)//224
    print("[INFO] loading label map & dataset ...")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # COCO 2017
    bbox = t.BboxParams(format='pascal_voc', min_visibility=0.2, label_fields=['parts'])
    tsfm_valid = t.Compose([t.Resize(size, size), t.Normalize(mean=mean, std=std), ToTensor()], bbox_params=bbox)
    tsfm_train = t.Compose([t.Resize(out_size, out_size), t.CenterCrop(size, size), t.Normalize(mean=mean, std=std),
                            ToTensor()], bbox_params=bbox)
    if 'FFoCat' in name:
        root_dir = f'Data/{name}'
        class_map = _build_food_labels_dict(root_dir)
        data_t, data_v = _get_food_data(class_map, root_dir, train=True), _get_food_data(class_map, root_dir, train=0)
    elif name == 'PASCAL':
        class_map = {
            'Bottle': ['Cap', 'Body'], 'Pottedplant': ['Pot', 'Plant'], 'Tvmonitor': ['Screen', 'Tvmonitor'],
            'Boat': ['Boat'], 'Chair': ['Chair'], 'Diningtable': ['Diningtable'], 'Sofa': ['Sofa'],
            'Bus': ['License_plate', 'Door', 'Wheel', 'Headlight', 'Bodywork', 'Mirror', 'Window'],
            'Car': ['License_plate', 'Door', 'Wheel', 'Headlight', 'Bodywork', 'Mirror', 'Window'],
            'Sheep': ['Torso', 'Tail', 'Muzzle', 'Neck', 'Eye', 'Horn', 'Leg', 'Ear', 'Head'],
            'Horse': ['Hoof', 'Torso', 'Muzzle', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
            'Dog': ['Torso', 'Muzzle', 'Nose', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
            'Bird': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Beak', 'Animal_Wing', 'Head'],
            'Cow': ['Torso', 'Muzzle', 'Tail', 'Horn', 'Eye', 'Neck', 'Leg', 'Ear', 'Head'],
            'Aeroplane': ['Stern', 'Engine', 'Wheel', 'Artifact_Wing', 'Body'],
            'Cat': ['Torso', 'Tail', 'Neck', 'Eye', 'Leg', 'Ear', 'Head'],
            'Motorbike': ['Wheel', 'Headlight', 'Saddle', 'Handlebar'],
            'Bicycle': ['Chain_Wheel', 'Saddle', 'Wheel', 'Handlebar'],
            'Person': ['Ebrow', 'Foot', 'Arm', 'Torso', 'Nose', 'Hair',
                'Hand', 'Neck', 'Eye', 'Leg', 'Ear', 'Head', 'Mouth'],
            'Train': ['Locomotive', 'Coach', 'Headlight'],
        }
        data_t, data_v = _get_pascal_data(class_map, train=True), _get_pascal_data(class_map, train=False)
    elif name == 'MonuMAI':
        class_map = {
            'Renaissance': ['fronton', 'fronton-curvo', 'serliana',
                            'arco-medio-punto', 'vano-adintelado', 'ojo-de-buey'],
            'Hispanic-Muslim': ['arco-herradura', 'arco-lobulado', 'dintel-adovelado'],
            'Gothic': ['arco-apuntado', 'arco-conopial', 'arco-trilobulado', 'pinaculo-gotico'],
            'Baroque': ['arco-medio-punto', 'vano-adintelado', 'ojo-de-buey', 'fronton-partido', 'columna-salomonica']
        }  # 'columna-salomonica' es nueva
        data_t, data_v = _get_monumai_data(class_map, train=True), _get_monumai_data(class_map, train=False)
    else:
        raise Exception('Dataset not implemented')
    valid = ShapImageDataset(data_v, class_map, tsfm_valid, device=device)
    train = ShapImageDataset(data_t, class_map, tsfm_train, device=device)
    # mean, std = get_mean_std(DataLoader(train, batch_size=100))
    # tsfm_train = T.Compose([T.RandomRotation(30), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), ToTensor()])
    # tsfm_train = t.Compose([t.Resize(256, 256), t.CenterCrop(224, 224), t.Normalize(mean=mean, std=std), ToTensor()],
    #                        bbox_params=t.BboxParams(format='pascal_voc', label_fields=['parts'], min_visibility=0.2))
    # train.tf = tsfm_train
    return train, valid


def test():
    from torch.utils.data import DataLoader
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.chdir('..')
    for db in ('PASCAL', 'FFoCat_tiny', 'MonuMAI'):
        data_train, data_tests = get_dataset(db, device=dev)
        a, b = data_train.detect[0], data_tests.classify[2:4]
        for n in (a, b):
            print(n[0][0].shape, type(n[0][-1]))
        for n in DataLoader(data_tests.classify, batch_size=2):
            print(n)
            break
        for n in DataLoader(data_train, batch_size=2, collate_fn=lambda batch: tuple(zip(*batch))):
            print(n)
            break
    exit(0)


if __name__ == '__main__':
    test()

"""
TODO - docs and type hints
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
    return df


def _get_pascal_data(cmap, root_dir='Data/LTN_ACM_SAC17/pascalpart_dataset/JPEGImages', train=False) -> pd.DataFrame:
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
    return df


def _get_monumai_data(cmap, root_dir='Data/OD-MonuMAI/MonuMAI_dataset/', train=False) -> pd.DataFrame:
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
    return df


class ShapImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, class_map: dict, transforms: t.Compose = ToTensor(), name='',
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.df, self.tf, self.cmap = dataframe[dataframe.is_part], transforms, class_map
        self.part_list = sorted(set(fea for ele in self.cmap.values() for fea in ele))
        self.class_list = sorted(self.cmap.keys())
        self.device, self.name = device, name

    def part_label_count(self, label_nums):
        # MultiLabelBinarizer(classes=self.part_list)  # Does not count how many of each part
        count = [0.]*len(self.part_list)
        for lbl in label_nums:
            count[lbl] += 1.
        return count

    def __len__(self):
        return self.df.img.unique().__len__()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        elif isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, np.int64):
            idx = [idx.tolist()]
        elif isinstance(idx, slice):
            step = idx.step if idx.step else 1
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else (self.__len__() - 1)
            idx = range(start, stop, step)

        tmp = self.tf.processors, self.tf.is_check_args
        if self.df.x0.iloc[0] < 0.:
            self.tf.is_check_args = False
            self.tf.processors = {}

        images, targets, clases, parts = [], [], [], []
        for i in idx:
            imgf = self.df.loc[[i], ['img']].values[0, 0]
            prts = self.df.loc[[i], 'part'].values.tolist()
            clas = self.df.loc[[i], ['class']].values[0, 0]
            boxs = self.df.loc[[i], ['x0', 'y0', 'x1', 'y1']].values

            aug = self.tf(image=np.array(Image.open(imgf).convert('RGB')), bboxes=boxs, parts=prts)
            img, boxs, prts = aug['image'], aug['bboxes'], aug['parts']

            boxs = torch.tensor(np.array(boxs)[:, :4], dtype=torch.float, device=self.device)
            area = (boxs[:, 3] - boxs[:, 1]) * (boxs[:, 2] - boxs[:, 0])
            prts = [self.part_list.index(p) for p in prts]
            targets.append({
                'iscrowd': torch.zeros((len(prts),), dtype=torch.long, device=self.device),
                'image_id': torch.tensor([i], dtype=torch.long, device=self.device),
                'labels': torch.tensor(prts, dtype=torch.long, device=self.device),
                'area': area, 'boxes': boxs,  # 'masks': None,
                'file_name': [imgf],
            })
            clases.append(self.class_list.index(clas))
            parts.append(self.part_label_count(prts))
            images.append(img)

        parts = torch.squeeze(torch.tensor(parts, dtype=torch.float).view(-1, len(self.part_list)))
        clases = torch.squeeze(torch.tensor(clases, dtype=torch.long).view(-1, 1))
        # clases = F.one_hot(classes, num_classes=len(self.classes))
        images = torch.squeeze(torch.stack(images))

        if self.df.x0.iloc[0] < 0.:
            self.tf.processors, self.tf.is_check_args = tmp
        return images.to(self.device), [targets, parts.to(self.device), clases.to(self.device)]


def get_dataset(name='FFoCat', size=224, device=torch.device('cpu')) -> [ShapImageDataset, ShapImageDataset]:
    def get_mean_std(datas):
        means, stds = [], []
        for img in datas:
            stds.append(torch.std(img))
            means.append(torch.mean(img))
        return torch.mean(torch.tensor(means)), torch.mean(torch.tensor(stds))

    out_size = (256*size)//224
    print("[INFO] loading label map & dataset ...")
    mean, std = [.485, .456, .406], [.229, .224, .225]  # COCO 2017
    bbox = t.BboxParams(format='pascal_voc', min_visibility=.2, label_fields=['parts'])
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
    valid = ShapImageDataset(data_v, class_map, tsfm_valid, name=name, device=device)
    train = ShapImageDataset(data_t, class_map, tsfm_train, name=name, device=device)
    # mean, std = get_mean_std(DataLoader(train, batch_size=100))
    # tsfm_train = t.Compose([t.Resize(256, 256), t.CenterCrop(224, 224), t.Normalize(mean=mean, std=std), ToTensor()],
    #                        bbox_params=t.BboxParams(format='pascal_voc', label_fields=['parts'], min_visibility=0.2))
    # train.tf = tsfm_train
    # # tsfm_train = T.Compose([T.RandomRotation(30), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), ToTensor()])
    return train, valid


def shap_collate_fn(bat):
    imgs = torch.squeeze(torch.stack([b[0] for b in bat]))
    tars = [b[1][0] for b in bat]
    lbls = torch.squeeze(torch.stack([b[1][1] for b in bat]))
    clss = torch.squeeze(torch.stack([b[1][2] for b in bat]))
    return imgs, (tars, lbls, clss)


def test():
    from torch.utils.data import DataLoader
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.chdir('..')
    for db in ('FFoCat_tiny', 'PASCAL', 'MonuMAI'):
        data_train, data_tests = get_dataset(db, device=dev)
        i, (t, l, c) = data_train[:2]
        print(i.shape, l, c)
        for tt in t:
            print(tt)
        for n in DataLoader(data_tests, batch_size=2, collate_fn=shap_collate_fn):
            for nn in n:
                print(nn)
            break
    exit(0)


if __name__ == '__main__':
    test()

"""
TODO - docs and type hints
"""
import os
import warnings

import torch
import numpy as np
import pandas as pd
import xml.etree.ElementTree as Et
import albumentations as t  # from torchvision import transforms as t

from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor

# from sklearn.preprocessing import MultiLabelBinarizer

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


def _get_food_data(cmap, root_dir='Data/FFoCat', train=False, unique_ingredient=True) -> pd.DataFrame:
    mode = 'train' if train else 'valid'
    data = {'img': [], 'class': [], 'boxes': []}
    used_recipes = set()
    overlap = 0
    for r, d, f in os.walk(os.path.join(root_dir, mode)):
        for dd in d:
            parts = tuple(sorted(cmap[dd]))
            if unique_ingredient and all(label in used_recipes for label in parts):  # 1+ unique ingredient per recipe
                overlap += 1
                continue
            elif not unique_ingredient and parts in used_recipes:  # unique recipes
                overlap += 1
                continue
            else:  # Only use recipes that don't completely overlap
                if unique_ingredient:
                    used_recipes.update(parts)
                else:
                    used_recipes.add(parts)
            boxs = [[-1, -1, -1, -1, label] for label in parts]
            for rr, _, ff in os.walk(os.path.join(r, dd)):
                for fff in ff:
                    data['img'].append(os.path.join(rr, fff))
                    data['boxes'].append(boxs)
                    data['class'].append(dd)
    df = pd.DataFrame(data)
    df = df.explode('boxes')
    df['is_part'] = True
    df[['x0', 'y0', 'x1', 'y1', 'part']] = df.boxes.tolist()
    return df.drop(columns=['boxes'])


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
                    boxs.append([box[0], box[1], box[2] + (box[0] == box[2]), box[3] + (box[1] == box[3]),
                                 prt, prt in part_list])  # TODO - better or move to DataLoader
                if prt in cmap:
                    groups.add(prt)
            if len(groups) < 2 and len(boxs):  # Imgs w/only 1 global/macro label
                data['boxes'].append(boxs)
                data['class'].append(cls)
                data['img'].append(img)
    df = pd.DataFrame(data)
    df = df.explode('boxes')
    df[['x0', 'y0', 'x1', 'y1', 'part', 'is_part']] = df.boxes.tolist()
    return df.drop(columns=['boxes'])


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
            boxs.append([box[0], box[1], box[2] + (box[0] == box[2]), box[3] + (box[1] == box[3]),
                         n.find('name').text])  # TODO - better
        data['boxes'].append(boxs)
    df = pd.DataFrame(data)
    df = df.explode('boxes')
    df['is_part'] = True
    df[['x0', 'y0', 'x1', 'y1', 'part']] = df.boxes.tolist()
    return df.drop(columns=['boxes'])


class ShapImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, class_map: dict, transforms: t.Compose = ToTensor(), name='',
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.df, self.tf, self.cmap = dataframe[dataframe.is_part], transforms, class_map
        self.part_list = sorted(set(fea for ele in self.cmap.values() for fea in ele))
        self.class_list = sorted(self.cmap.keys())
        self.device, self.name = device, name
        self.count = False  # TODO - Use count
        # MultiLabelBinarizer(classes=self.part_list)  # Does not count how many of each part
        self._indices = self.df.index.unique()

    def part_label_count(self, label_nums):
        count = [0.]*len(self.part_list)
        if self.count:
            for lbl in label_nums:
                count[lbl] += 1.
        else:
            for lbl in label_nums:
                count[lbl] = 1.
        return count

    def __len__(self):
        # self.df.index[-1] + 1
        return self._indices.__len__()

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
            df_local = self.df.loc[[self._indices[i]]]
            imgf = df_local.loc[:, ['img']].values[0, 0]
            prts = df_local.loc[:, 'part'].values.tolist()
            clas = df_local.loc[:, ['class']].values[0, 0]
            boxs = df_local.loc[:, ['x0', 'y0', 'x1', 'y1']].values

            try:
                with open(imgf, 'rb') as img_f:
                    aug = self.tf(image=np.array(Image.open(img_f).convert('RGB')), bboxes=boxs, parts=prts)
            except UserWarning as e:
                # print(imgf)
                continue
                # raise e

            img, boxs, prts = aug['image'], aug['bboxes'], aug['parts']

            if boxs and self.df.x0.iloc[0] >= 0.:
                boxs = torch.tensor(np.array(boxs)[:, :4], dtype=torch.float, device=self.device)
                area = (boxs[:, 3] - boxs[:, 1]) * (boxs[:, 2] - boxs[:, 0])  # (y1 - y0) * (x1 - x0)
                if any(torch.le(area, 0.)):
                    breakpoint()
            else:
                boxs = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                area = torch.zeros(0, dtype=torch.float32, device=self.device)

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
            # print(imgf)

        parts = torch.squeeze(torch.tensor(parts, dtype=torch.float).view(-1, len(self.part_list)))
        clases = torch.squeeze(torch.tensor(clases, dtype=torch.long).view(-1, 1))
        # clases = torch.nn.functional.one_hot(clases, num_classes=len(self.class_list))
        images = torch.squeeze(torch.stack(images)) if images else torch.empty(0)
        if images.size()[0] == 0:
            print(parts.size(), clases.size(), targets)

        if self.df.x0.iloc[0] < 0.:
            self.tf.processors, self.tf.is_check_args = tmp
        return images.to(self.device), [targets, parts.to(self.device), clases.to(self.device)]


def get_dataset(name='FFoCat', size=224, simple_map=False, custom=None,
                device=torch.device('cpu')) -> [ShapImageDataset, ShapImageDataset]:
    def get_mean_std(datas):
        means, stds = [], []
        for img in datas:
            stds.append(torch.std(img))
            means.append(torch.mean(img))
        return torch.mean(torch.tensor(means)), torch.mean(torch.tensor(stds))

    out_size = (256*size)//224
    print(f'[INFO] loading label map & dataset ({name})...')
    mean, std = [.485, .456, .406], [.229, .224, .225]  # COCO 2017
    bbox = t.BboxParams(format='pascal_voc', min_visibility=.2, label_fields=['parts'])
    tsfm_valid = t.Compose([t.Resize(size, size), t.Normalize(mean=mean, std=std), ToTensor()], bbox_params=bbox)
    tsfm_train = t.Compose([t.Resize(out_size, out_size), t.CenterCrop(size, size), t.Normalize(mean=mean, std=std),
                            ToTensor()], bbox_params=bbox)
    options = {}
    if 'FFoCat' in name:
        options['root_dir'] = f'Data/' + (name[:-5] if name[-5:] == '_test' else name)
        class_map = _build_food_labels_dict(options['root_dir'])
        get_data = _get_food_data
    elif 'PASCAL' in name:
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
        get_data = _get_pascal_data
    elif 'MonuMAI' in name:
        class_map = {
            'Renaissance': ['fronton', 'fronton-curvo', 'serliana',
                            'arco-medio-punto', 'vano-adintelado', 'ojo-de-buey'],
            'Hispanic-Muslim': ['arco-herradura', 'arco-lobulado', 'dintel-adovelado'],
            'Gothic': ['arco-apuntado', 'arco-conopial', 'arco-trilobulado', 'pinaculo-gotico'],
            'Baroque': ['arco-medio-punto', 'vano-adintelado', 'ojo-de-buey', 'fronton-partido', 'columna-salomonica']
        }  # 'columna-salomonica' es nueva
        get_data = _get_monumai_data
    elif 'custom' in name:
        class_map = simple_map
        get_data = custom  # lambda x: x  # should be list of filenames and their labels directly,... probably
        raise NotImplementedError('Custom dataset not implemented')
    else:
        raise Exception('Unknown dataset')

    if simple_map is True:  # Original code does this
        used = set()
        for clas, parts in class_map.items():
            class_map[clas] = [p for p in parts if parts not in used]
            used.update(parts)
    data_t, data_v = get_data(class_map, **options, train=True), get_data(class_map, **options, train=False)

    if name[-5:] == '_test':
        data_t = data_t.loc[np.random.choice(data_t.index[-1] + 1, 32, False)]
        data_v = data_v.loc[np.random.choice(data_v.index[-1] + 1,  8, False)]
    valid = ShapImageDataset(data_v, class_map, tsfm_valid, name=name, device=device)
    train = ShapImageDataset(data_t, class_map, tsfm_train, name=name, device=device)
    # mean, std = get_mean_std(DataLoader(train, batch_size=100))
    # tsfm_train = t.Compose([t.Resize(256, 256), t.CenterCrop(224, 224), t.Normalize(mean=mean, std=std), ToTensor()],
    #                        bbox_params=t.BboxParams(format='pascal_voc', label_fields=['parts'], min_visibility=0.2))
    # train.tf = tsfm_train
    # # tsfm_train = T.Compose([T.RandomRotation(30), T.RandomResizedCrop(224), T.RandomHorizontalFlip(), ToTensor()])
    return train, valid


def shap_collate_fn(bat):
    imgs = torch.stack([b[0] for b in bat if b[0].size()[0] > 0])
    tars = [b[1][0][0] for b in bat if b[0].size()[0] > 0]
    lbls = torch.stack([b[1][1] for b in bat if b[0].size()[0] > 0])
    clss = torch.stack([b[1][2] for b in bat if b[0].size()[0] > 0])
    # print(imgs.shape, lbls.shape, clss.shape)
    return imgs, (tars, lbls, clss)


def move_to_device(data, device):
    imgs, (targets, parts, clases) = data
    for tar in targets:
        for key, val in tar.items():
            if torch.is_tensor(val):
                tar[key] = val.to(device)
    return imgs.to(device), (targets, parts.to(device), clases.to(device))


def test():  # TODO - move tests to their own directory - Also, probably should be UnitTest
    from torch.utils.data import DataLoader
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.chdir('../..')
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


if __name__ == '__main__':
    test()

'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
from PIL import Image
import os
import numpy as np
import torch
import torchvision
from torchvision.io import write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T


__all__ = ('CustomDataset', 'Repeat')

mvtec_class_names = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

btad_class_names = ['01','02','03']

# repeat a dataset until the length of dataset reaches "new_length"
class Repeat(Dataset):
    def __init__(self, org_dataset, new_length):
        self.org_dataset = org_dataset
        self.org_length = len(self.org_dataset)
        self.new_length = new_length

    def __len__(self):
        return self.new_length

    def __getitem__(self, idx):
        return self.org_dataset[idx % self.org_length]

# construct dataset 
class CustomDataset(Dataset):
    def __init__(self, img_size, data_path, class_name, dataset, norm_mean, norm_std, is_train=True):
        self.img_size = img_size
        self.dataset_path = os.path.join(data_path, dataset)
        self.class_name = class_name
        self.is_train = is_train

        if dataset == 'mvtec':
            assert class_name in mvtec_class_names, 'class_name of {}: {}, should be in {}'.format(dataset, class_name, mvtec_class_names)
            self.x, self.y, self.mask= self.load_mvtec_dataset_folder(self.class_name)
        if dataset == 'btad':
            assert class_name in btad_class_names, 'class_name of {}: {}, should be in {}'.format(dataset, class_name, btad_class_names)
            self.x, self.y, self.mask= self.load_btad_dataset_folder(self.class_name)

        # mask
        self.transform_mask = T.Compose([
            T.Resize(self.img_size, torchvision.transforms.InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(norm_mean, norm_std)])

    def __getitem__(self, idx):
        x, y, mask= self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)
        np_x = np.squeeze(np.array(x)) 
        if len(np_x.shape)==2:  # handle greyscale classes
            x = np.expand_dims(np_x, axis=2)
            x = np.concatenate([x, x, x], axis=2)
            x = Image.fromarray(x.astype('uint8'))
        else:
            pass
        #
        org_x = x

        self.transform_x = T.Compose([
            T.Resize(self.img_size, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor()])

        aug_x= self.transform_x(org_x)
        x = self.normalize(aug_x)
        #
        if y == 0:
            mask = torch.zeros([1, self.img_size[0], self.img_size[1]])
        else:
            if os.path.isfile(mask)==True:
                mask = Image.open(mask)
                mask = self.transform_mask(mask)
                mask = torch.mean(mask, dim=0, keepdim=True)
            else:
                print(self.x[idx])
                print(y)
                raise KeyboardInterrupt

        mask = (mask>0).type(torch.long)
        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_mvtec_dataset_folder(self, class_name):
        x, y, mask= [], [], []

        phase = 'train' if self.is_train else 'test'
        img_dir = os.path.join(self.dataset_path, class_name, phase)
        gt_dir = os.path.join(self.dataset_path, class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png') or f.endswith('.JPEG')])

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                x.extend(img_fpath_list)
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                x.extend(img_fpath_list)
            if len(x)!=len(y):
                print(len(x))
                print(len(y))
                print(img_type)
                raise KeyboardInterrupt
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

    def load_btad_dataset_folder(self, class_name):
        x, y, mask= [], [], []

        phase = 'train' if self.is_train else 'test'
        img_dir = os.path.join(self.dataset_path, class_name, phase)
        gt_dir = os.path.join(self.dataset_path, class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png') or f.endswith('.bmp')])

            # load gt labels
            if img_type == 'ok':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                x.extend(img_fpath_list)
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png') if os.path.isfile(os.path.join(gt_type_dir, img_fname + '.png')) ==True else os.path.join(gt_type_dir, img_fname + '.bmp') for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                x.extend(img_fpath_list)
            if len(x)!=len(y):
                print(len(x))
                print(len(y))
                print(img_type)
                raise KeyboardInterrupt
        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)


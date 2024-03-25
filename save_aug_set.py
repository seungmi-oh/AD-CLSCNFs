import os, random, copy, cv2
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision.io import write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from custom_datasets import * 
from utils import *
from config import get_args

c = get_args()
c.img_size = (c.input_size, c.input_size)  # HxW format
c.crp_size = (c.input_size, c.input_size)  # HxW format
c.img_dims = [3] + list(c.img_size)

save_dir_single_samples = make_aug_dataset_path(c, c.aug_sample_path)
save_dir_single = make_aug_dataset_path(c, c.data_aug_path) 

if c.dataset == 'mvtec':
    class_name_list = mvtec_class_names
    total_type_list = ['defect', 'good']
if c.dataset == 'btad':
    class_name_list = btad_class_names
    total_type_list = ['ko', 'ok']

for cl in class_name_list:
    c.class_name = cl
    print(cl)
     
    # save synthetic defect samples
    train_dataset = GenerateSyntheticTrainDataset(c.data_path, c.class_name, c.dataset, c.norm_mean, c.norm_std, 1.0, c.use_in_domain_data, resize_shape = c.img_size, for_check=True) 
    sample_num = min(50, len(train_dataset))
    print(f'save {sample_num} samples')
    save_dir = os.path.join(save_dir_single_samples, c.dataset, cl)
    save_aug_samples(train_dataset, sample_num, save_dir)

    # save synthetic defect dataset
    train_dataset = GenerateSyntheticTrainDataset(c.data_path, c.class_name, c.dataset, c.norm_mean, c.norm_std, 1.0, c.use_in_domain_data, resize_shape = c.img_size) 
    total_num = len(train_dataset)
    shuffle_indices = list(range(total_num))
    random.shuffle(shuffle_indices)
    j=0
    for defect_idx in range(len(total_type_list)):
        defect_name = total_type_list[defect_idx]
        if defect_name == 'good' or defect_name == 'ok':
            train_dataset = GenerateSyntheticTrainDataset(c.data_path, c.class_name, c.dataset, c.norm_mean, c.norm_std, 0, c.use_in_domain_data, resize_shape = c.img_size) 
        else:
            train_dataset = GenerateSyntheticTrainDataset(c.data_path, c.class_name, c.dataset, c.norm_mean, c.norm_std, 1.0, c.use_in_domain_data, resize_shape = c.img_size) 
        sample_num = int(len(train_dataset)/len(total_type_list))
        init_idx = defect_idx*sample_num
        fin_idx = (defect_idx+1)*(sample_num)
        if defect_idx ==len(total_type_list)-1:
            idx_list = shuffle_indices[init_idx:]
        else:
            idx_list = shuffle_indices[init_idx:fin_idx]
        std = torch.FloatTensor(c.norm_std)
        mean = torch.FloatTensor(c.norm_mean)
        std = torch.unsqueeze(torch.unsqueeze(std, -1), -1)
        mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1)

        print(f'{defect_name} data: {len(idx_list)} samples')
        save_dir = os.path.join(save_dir_single, c.dataset, cl)
        save_single_aug_set(idx_list, train_dataset, std, mean, save_dir, defect_name, j, c.dataset)




import os, cv2
import numpy as np
import torch
from utils import makedirs, t2np
from custom_datasets import * 
from config import get_args

# convert pytorch tensor or numpy array to cv2 format 
def t2cv2(x):
    if isinstance(x, np.ndarray)==True:
        x = cv2.cvtColor(np.uint8(x), cv2.COLOR_RGB2BGR)
    else:
        x = x.type(torch.uint8)
        x = t2np(x)
        if (len(x.shape)==2) or (x.shape[0]==1):
            x = np.transpose([np.squeeze(x)]*3, (1,2,0))
        else:
            x = np.transpose(x, (1,2,0))
        x = cv2.cvtColor(np.uint8(x), cv2.COLOR_RGB2BGR)
    return x

# save augmentation samples to check the synthetic defect generation process
def save_aug_samples(train_dataset, sample_num, save_dir):
    for idx in range(sample_num):
        viz_list = train_dataset[idx]
        viz_list = list(viz_list)
        new_viz_list = []
        for j in range(len(viz_list)):
            viz_data = viz_list[j]
            margin = np.uint8(np.ones((viz_data.shape[0],10,3))*255)
            new_viz_list.append(t2cv2(viz_data))
            new_viz_list.append(margin)

        save_img = np.uint8(np.concatenate(new_viz_list[:-1],1))

        makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, f'sample{idx:02d}.png'), save_img)

# save synthetic defect dataset to evaluate networks
def save_single_aug_set(idx_list, train_dataset, std, mean, save_dir, defect_name, j, dataset):  
    file_list = train_dataset.image_paths
    for idx in idx_list:
        x,y, mask= train_dataset[idx]
        x = (x*std+mean)*255
        x = t2cv2(x)

        mask = t2cv2(mask*255)

        file_name = os.path.basename(file_list[idx])
        makedirs(os.path.join(save_dir, 'test', defect_name))
        makedirs(os.path.join(save_dir, 'ground_truth', defect_name))

        if dataset =='mvtec':
            if np.sum(mask)>0:
                cv2.imwrite(os.path.join(save_dir, 'test', defect_name, f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), x)
                cv2.imwrite(os.path.join(save_dir, 'ground_truth', defect_name, f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}_mask.png'), mask)
            else:
                makedirs(os.path.join(save_dir, 'test', 'good'))
                cv2.imwrite(os.path.join(save_dir, 'test', 'good', f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), x)
        if dataset =='btad':
            if np.sum(mask)>0:
                cv2.imwrite(os.path.join(save_dir, 'test', defect_name, f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), x)
                cv2.imwrite(os.path.join(save_dir, 'ground_truth', defect_name, f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), mask)
            else:
                makedirs(os.path.join(save_dir, 'test', 'ok'))
                cv2.imwrite(os.path.join(save_dir, 'test', 'ok', f'{file_name[:-4]}_sample{j*len(train_dataset)+idx:04d}.png'), x)

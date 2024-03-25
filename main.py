'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
from __future__ import print_function
import os, random, time, math, datetime, imgaug
import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from config import get_args
from utils import makedirs, make_model_path
from train_methods import *
from custom_datasets import * 
import shutil


# for repeatability 
def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    imgaug.seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

# cache the input configure values
def cache_test_args(c, config_list):
    temp_config_list = []
    for arg_name in config_list:
        val = getattr(c, arg_name)
        temp_config_list.append(val)
    return temp_config_list

# replace the values saved in the state dict of the trained model to the input configures for inference
def redefine_test_args(c, config_list, temp_config_list):
    for i in range(len(config_list)):
        arg_name = config_list[i]
        setattr(c, arg_name, temp_config_list[i])

def main(c):
    # generate path to save or load models
    if c.train_type == 'nf_only':
        pl_str = [str(pl) for pl in c.pool_layers]
        pool_layers = '-'.join(pl_str)
        c.train_type = f'{c.train_type}_pl{pool_layers}'


    if c.is_train==True:
        c.model_dir = make_model_path(c)
        makedirs(c.model_dir)
        if os.path.isdir(f'{c.model_dir}/codes')==True:
            shutil.rmtree(f'{c.model_dir}/codes')
        else:
            pass
        shutil.copytree('./',f'{c.model_dir}/codes')
        c.run_date = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    else:
        c.model_dir = make_model_path(c)
        weight_dir = os.path.join(c.model_dir, 'weights')
        if c.infer_type == 'fe_only' or ('base' in c.infer_type)==True:
            sub_path = c.infer_type
        else:
            pl_str = [str(pl) for pl in c.pool_layers]
            pool_layers = '-'.join(pl_str)
            sub_path = f'nf_only_pl{pool_layers}'
        model_files_=os.listdir(weight_dir)
        for model_file_ in model_files_:
            if (sub_path in model_file_)==True:
                model_path=os.path.join(weight_dir, model_file_)
            else: 
                pass
        if 'model_path' not in locals():
            print(f'There is no trained files (train type: {sub_path})')
            raise KeyboardInterrupt

        # configure list to replace values saved in the state_dict of trained model to input values for inference
        config_list = ['batch_size', 'workers', 'train_type', 'class_name', 'is_close', 'is_open', 'is_k_disk', 'feat_avg_topk', 'k_size', 'data_path', 'data_aug_path', 'th_manual', 'infer_type', 'is_train', 'test_data_type', 'model_dir', 'pro', 'viz', 'w_fe', 'get_best_w_fe']

        if c.seed == None:
            pass
        else:
            config_list.extend(['seed'])

        temp_config_list = cache_test_args(c, config_list)
        state = torch.load(model_path)
        c = state['args']

        redefine_test_args(c, config_list, temp_config_list)
        del state

    # image
    c.img_size = (c.input_size, c.input_size)  # HxW format
    c.crp_size = (c.input_size, c.input_size)  # HxW format
    c.img_dims = [3] + list(c.img_size)

    num_class =1 
    if c.aug_ratio_train >0:
        num_class += 1 
    else:
        pass

    c.num_class = num_class

    # set hyper parameters for scheduling learning rate 
    if c.train_type =='fe_only':
        total_epoch = c.meta_epochs+c.freeze_enc_epochs
    else:
        total_epoch = c.meta_epochs
    c.lr_decay_epochs = [int(c.lr_decay_epochs_percentage[i]*total_epoch) for i in range(len(c.lr_decay_epochs_percentage))]
    print('LR schedule: {}'.format(c.lr_decay_epochs))
    if c.lr_warm:
        c.lr_warmup_from = c.lr/10.0
        if c.lr_cosine:
            eta_min = c.lr * (c.lr_decay_rate ** 3)
            c.lr_warmup_to = eta_min + (c.lr - eta_min) * (
                    1 + math.cos(math.pi * c.lr_warm_epochs / total_epoch)) / 2
        else:
            c.lr_warmup_to = c.lr

    # set device 
    os.environ['CUDA_VISIBLE_DEVICES'] = c.gpu
    if c.seed ==None:
        c.seed = int(time.time())
    init_seeds(seed=c.seed)
    c.use_cuda = not c.no_cuda and torch.cuda.is_available()
    c.device = torch.device("cuda" if c.use_cuda else "cpu")

    c_dict = vars(c)
    dict_string=''
    for k in c_dict.keys():
        dict_string = f'{dict_string}\n{k}:{c_dict[k]}'

    # train or test  
    if c.is_train ==True:
        # save configure file
        with open(os.path.join(c.model_dir, f'{c.train_type}_config_list.txt'), 'w') as f:
            f.write(dict_string)

        if 'base' in c.train_type:
            log_txt_path = train_CNFs_pretrain.run(c)
        elif 'fe_only' == c.train_type:
            log_txt_path = finetune_enc.run(c)
        else:
            log_txt_path = train_CNFs_finetune.run(c)
    else:
        if 'joint' in c.infer_type:
            c.add_fe_anomaly =True
        else:
            c.add_fe_anomaly =False	

        if 'base' in c.infer_type:
            log_txt_path = train_CNFs_pretrain.run(c)
        elif 'fe_only' == c.infer_type:
            log_txt_path = finetune_enc.run(c)
        elif 'not_sharing' in c.infer_type:
            log_txt_path = eval_not_sharing_enc.test(c)
        else:
            log_txt_path = train_CNFs_finetune.run(c)

    # print configuration and results
    if c.is_train==False:
        txt_file = open(log_txt_path, 'r')
        log_str = txt_file.read()
        txt_file.close()
        print(f'[{c.dataset.upper()}] {c.class_name} inference process is finish!', dict_string+'\n'+log_str) 
    else:
        log_str = ''
        print(f'[{c.dataset.upper()}] {c.class_name} training process is finish!', dict_string+'\n'+log_str) 


if __name__ == '__main__':
    c = get_args()
    main(c)


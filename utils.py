'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import os, cv2, copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

_GCONST_ = -0.9189385332046727 # ln(sqrt(2*pi))

# monitoring validation results
class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = 0.0
        self.last = 0.0

    def update(self, score, epoch, txt_file, print_score=True):
        self.last = score
        save_weights = False
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            save_weights = True
        if print_score:
            self.print_score(txt_file)
        
        return save_weights

    def print_score(self, txt_file):
        print('{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))
        txt_file.write('\n{:s}: \t last: {:.2f} \t max: {:.2f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.detach().cpu().numpy() if tensor is not None else None

# calculate log-likelihood for input feature vectors 
def get_logp(C, z, logdet_J, m):
    m = torch.squeeze(m, dim=1)
    logp = C * _GCONST_ - 0.5*torch.sum(z**2, 1) + logdet_J # shape:(B,H,W)
    defect_logp = logp.clone() 
    defect_logp[m==1]=0
    good_logp = logp.clone() 
    good_logp[m==-1] =0
    return logp, defect_logp, good_logp


# normalize to the range between 0 and 1
def rescale(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-7)


# make directory if it does not exist 
def makedirs(dir_name):
    if os.path.isdir(dir_name)==False:
        os.makedirs(dir_name)


def compare_models(model_1, model_2, txt_file, return_output = False, verbose =False):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                if verbose==True:
                    print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
        txt_file.write('\nModels match perfectly! :)')
    if return_output==True:
        print(models_differ)
        return models_differ

# for measuring inference time
def time_measure():
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)  
    return starter, ender

# write results in csv files
def write_csv(results:dict, csv_path, class_name):
    keys = list(results.keys())
    df_result = pd.DataFrame(results, index = [class_name])
    df_result.to_csv(csv_path, header=True, float_format='%.2f')


# calculate model size 
def cal_model_size_MB(model_list, result_dict):
    param_size = 0
    buffer_size = 0
    for model in model_list: 
        if isinstance(model, list):
            for model_ in model:
                for param in model_.parameters():
                    param_size += param.nelement() * param.element_size()
                for buffer in model_.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
        else:
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()

    size_MB = (param_size + buffer_size) / 1024**2
    result_dict['model_size (MB)'] = size_MB
    print(f'model size: {size_MB:.3f} MB')


# make a path to save models according to configures 
def make_model_path(c):
    skip_ = 'skip' if c.skip_connection ==True else 'no-skip'
    dec_layer = 'same-with-nf' if c.set_dec_dims_nf ==True else ''

    data_info = 'clean'

    if 'cls' in c.loss_type:
        if c.aug_ratio_train>0:
            data_info = f'{data_info}-synthetic_DRAEM_{int(c.aug_ratio_train*100):03d}'
        else:
            pass
    else:
        pass

    train_method = f'separate-{c.loss_type}'

    if c.pretrained ==True:
        train_method = f'{train_method}-pretrained'
    else:
        train_method = f'{train_method}-scratch'

    train_method = f'{train_method}-evalBN'

    if c.skip_connection ==True and c.set_dec_dims_nf ==True:
        c.model_name = "{}-{}-{}_{}_{}_{}_cb{}".format(data_info, train_method, c.enc_arch, skip_, dec_layer, c.nf_arch, c.coupling_blocks)
    else:
        c.model_name = "{}-{}-{}_{}_{}_cb{}".format(data_info, train_method, c.enc_arch, skip_, c.nf_arch, c.coupling_blocks)

    model_dir = os.path.join(c.model_path, c.dataset, c.class_name, f'inp_{c.input_size}', c.model_name, f'run_{c.run_name}')
    return model_dir


# make a path to save synthetic defect dataset or samples  
def make_aug_dataset_path(c, aug_set_root):
    if c.use_in_domain_data == True:
        aug_set_path = os.path.join(aug_set_root, 'with_in_domain')
    else:
        aug_set_path = os.path.join(aug_set_root, 'without_in_domain')
    return aug_set_path

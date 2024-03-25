'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import os, math
import numpy as np
import torch

__all__ = ('save_results', 'save_weights', 'load_weights', 'adjust_learning_rate', 'warmup_learning_rate')

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# save test results for the last epoch
def save_results(det_roc_obs, seg_roc_obs, seg_pro_obs, model_dir, class_name, run_date, data_type = 'real'):
    result = '{:.2f},{:.2f},{:.2f} \t\tfor {:s}/{:s}/{:s} at epoch {:d}/{:d}/{:d} for {:s}\n'.format(
        det_roc_obs.max_score, seg_roc_obs.max_score, seg_pro_obs.max_score,
        det_roc_obs.name, seg_roc_obs.name, seg_pro_obs.name,
        det_roc_obs.max_epoch, seg_roc_obs.max_epoch, seg_pro_obs.max_epoch, class_name)
    result_dir = os.path.join(model_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fp = open(os.path.join(result_dir, f'{data_type}-{run_date}.txt'), "w")
    fp.write(result)
    fp.close()


def save_weights(c, model, model_dir):
    weight_dir = os.path.join(model_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    # CFlow-AD (CNF networks with the pretrained feature extractor)
    if 'base' in c.train_type:
        encoder = model[0]
        nfs = model[-1]
        state = {'encoder_state_dict': encoder.state_dict(),
                 'nf_state_dict': [nf.state_dict() for nf in nfs],
                 'args': c}
    # the proposed method
    else:
        encoder = model[0]
        decoder = model[1]
        if c.train_type == 'fe_only':
            state = {'encoder_state_dict': encoder.state_dict(),
                     'decoder_state_dict': decoder.state_dict(),
                     'args': c}
        elif 'nf' in c.train_type:
            nfs = model[-1]
            state = {'encoder_state_dict': encoder.state_dict(),
                     'nf_state_dict': [nf.state_dict() for nf in nfs],
                     'args': c}
        else:
            raise NotImplementedError('{} is not supported train type!'.format(c.train_type))
    filename = f'{c.train_type}_{c.run_date}.pt'
    path = os.path.join(weight_dir, filename)
    torch.save(state, path)
    print('Saving weights to {}'.format(path))


# load weights according to inference type
def load_weights(c, model, infer_type):
    weight_dir = os.path.join(c.model_dir, 'weights')
    if infer_type == 'fe_only' or ('base' in infer_type)==True:
        sub_path = infer_type
    else:
        pl_str = [str(pl) for pl in c.pool_layers]
        pool_layers = '-'.join(pl_str)
        sub_path = f'nf_only_pl{pool_layers}'

    if os.path.exists(weight_dir)==True:
        model_files_=os.listdir(weight_dir)
        for model_file_ in model_files_:
            if (sub_path in model_file_)==True:
                model_path=os.path.join(weight_dir, model_file_)
            else: 
                pass
    if 'model_path' in locals():
        state = torch.load(model_path)
        c.num_class = state['args'].num_class
        if 'encoder_state_dict' in state:
            encoder = model[0]
            encoder.load_state_dict(state['encoder_state_dict'], strict=True)
        if 'nf_state_dict' in state:
            nfs = model[-1]
            nfs = [nf.load_state_dict(state, strict=True) for nf, state in zip(nfs, state['nf_state_dict'])]
        if 'decoder_state_dict' in state:
            decoder = model[1]
            decoder.load_state_dict(state['decoder_state_dict'], strict=True)
        print('Loading weights from {}'.format(model_path))
        return state
    else:
        print(f'No {infer_type} weights from {weight_dir}')
        return []


# learning rate scheduler
def adjust_learning_rate(c, optimizer, epoch, total_epoch):
    lr = c.lr
    if c.lr_cosine:
        eta_min = lr * (c.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / total_epoch)) / 2
    else:
        steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
        if steps > 0:
            lr = lr * (c.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# warm up learning rate scheduler
def warmup_learning_rate(c, epoch, batch_id, total_batches, optimizer):
    if c.lr_warm and epoch < c.lr_warm_epochs:
        p = (batch_id + epoch * total_batches) / \
            (c.lr_warm_epochs * total_batches)
        lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #
    for param_group in optimizer.param_groups:
        lrate = param_group['lr']
    return lrate

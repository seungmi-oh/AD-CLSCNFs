'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import torch
import math
from torch import nn
from custom_models import *
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm

# get positional embedding vector
def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P

# small network to obtain scale and shift parameters of CNF networks
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


# CNF network
def freia_cflow_head(c, n_feat):
    n_cond = c.condition_vec
    coder = Ff.SequenceINN(n_feat)
    print('CNF coder:', n_feat)
    for k in range(c.coupling_blocks):
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(n_cond,), subnet_constructor=subnet_fc, affine_clamping=c.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


def load_nf_arch(c, dim_in):
    if c.nf_arch == 'freia-cflow':
        nf = freia_cflow_head(c, dim_in)
    else:
        raise NotImplementedError('{} is not supported NF!'.format(c.nf_arch))
    return nf


# forward hook to get feature maps of decoder
dec_activation = {}
def get_dec_activation(name):
    def hook(model, input, output):
        dec_activation[name] = output.detach()
    return hook

# construct a decoder network 
def load_decoder_arch(c, dec_dims):
    if c.set_dec_dims_nf ==True:
        skip_layers_ = c.pool_layers
    else:
        skip_layers_ = list(range(len(c.dec_dims_fe)))

    if c.enc_arch == 'resnet18': 
        if c.loss_type =='cls':
            dec = resnet18_dec(num_classes = c.num_class, skip_layers = skip_layers_,final_activation ='relu', skip_connection=c.skip_connection) 
        elif c.loss_type =='reg':
            dec = resnet18_dec(num_classes = 3, skip_layers = skip_layers_, skip_connection=c.skip_connection, mean = c.norm_mean, std = c.norm_std) 
        elif 'smooth' in c.loss_type:
            dec = resnet18_dec(num_classes = c.num_class-1, skip_layers = skip_layers_, skip_connection=c.skip_connection) 
        else:
            raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))
    elif c.enc_arch == 'wide_resnet50_2':
        if c.loss_type =='cls':
            dec = wide_resnet50_2_dec(num_classes = c.num_class, skip_layers = skip_layers_,final_activation ='relu', skip_connection=c.skip_connection) 
        elif c.loss_type =='reg':
            dec = wide_resnet50_2_dec(num_classes = 3, skip_layers = skip_layers_, skip_connection=c.skip_connection, mean = c.norm_mean, std = c.norm_std) 
        elif 'smooth' in c.loss_type:
            dec = wide_resnet50_2_dec(num_classes = c.num_class-1, skip_layers = skip_layers_, skip_connection=c.skip_connection) 
        else:
            raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))
    else:
        pass

    for name, ch in dec.named_modules():
        if isinstance(ch, nn.Conv2d)==True:
            if ch.padding[0]>0 and ch.padding[1]>0:
                ch.padding_mode = 'reflect'
            elif isinstance(ch, nn.BatchNorm2d)==True:
                ch.track_running_stats =True
            else:
                pass

    dec_pool_layers = list()
    if 2 in skip_layers_:
        dec.uplayer1.register_forward_hook(get_dec_activation('dec_layer2'))
        dec_pool_layers.append('dec_layer2')
    if 1 in skip_layers_:
        dec.uplayer2.register_forward_hook(get_dec_activation('dec_layer1'))
        dec_pool_layers.append('dec_layer1')
    if 0 in skip_layers_:
        dec.uplayer3.register_forward_hook(get_dec_activation('dec_layer0'))
        dec_pool_layers.append('dec_layer0')

    dec_pool_dims = list()
    if 2 in skip_layers_:
        dec_pool_dims.append(dec_dims[2][0])
    if 1 in skip_layers_:
        dec_pool_dims.append(dec_dims[1][0])
    if 0 in skip_layers_:
        dec_pool_dims.append(dec_dims[0][0])
    return dec, dec_pool_layers, dec_pool_dims


# forward hook to get input feature maps of CNF networks 
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# construct an encoder network 
def load_encoder_arch(c, L=[0,1,2,3]):
    pool_dims = list()
    pool_layers = ['layer'+str(i) for i in L]
    pool_cnt = 0
    if 'resnet' in c.enc_arch:
        if   c.enc_arch == 'resnet18':
            encoder = resnet18(pretrained=c.pretrained, progress=True)
        elif c.enc_arch == 'wide_resnet50_2':
            encoder = wide_resnet50_2(pretrained=c.pretrained, progress=True)
        else:
            raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
        for name, ch in encoder.named_modules():
            if isinstance(ch, nn.Conv2d)==True:
                if ch.padding[0]>0 and ch.padding[1]>0:
                    ch.padding_mode = 'reflect' 
                else:
                    pass
            elif isinstance(ch, nn.BatchNorm2d)==True:
                ch.track_running_stats =False
            else:
                pass
        #
        if 0 in L:
            encoder.layer1.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer1[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer1[-1].conv2.out_channels)
            pool_cnt += 1
        if 1 in L:
            encoder.layer2.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer2[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer2[-1].conv2.out_channels)
            pool_cnt += 1
        if 2 in L:
            encoder.layer3.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer3[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer3[-1].conv2.out_channels)
            pool_cnt += 1
        if 3 in L:
            encoder.layer4.register_forward_hook(get_activation(pool_layers[pool_cnt]))
            if 'wide' in c.enc_arch:
                pool_dims.append(encoder.layer4[-1].conv3.out_channels)
            else:
                pool_dims.append(encoder.layer4[-1].conv2.out_channels)
            pool_cnt += 1
    else:
        raise NotImplementedError('{} is not supported architecture!'.format(c.enc_arch))
    return encoder, pool_layers, pool_dims

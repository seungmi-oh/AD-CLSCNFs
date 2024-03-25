'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import os, time, gc, copy
import numpy as np
import torch
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
from .evaluate import *
from model import load_nf_arch, load_encoder_arch, load_decoder_arch, activation, dec_activation, positionalencoding2d
from utils import *
from custom_datasets import *
from custom_models import *


# train CNF networks with the finetuned feature extractor
def train_meta_epoch(c, epoch, loader, model, optimizers, pool_layers, fe_loss_fn, txt_file):
    log_theta = torch.nn.LogSigmoid()
    P = c.condition_vec
    N = c.p_bs
    I = len(loader)
    iterator = iter(loader)

    encoder, decoder,  nfs = model
    encoder = encoder.eval()
    decoder = decoder.eval()
    nfs = [nf.train() for nf in nfs]

    # update learning rate
    adjust_learning_rate(c, optimizers, epoch, c.meta_epochs)

    for sub_epoch in range(c.sub_epochs):
        train_loss = 0.0
        train_defect_nf_loss = 0.0
        train_good_nf_loss = 0.0
        train_count = 0
        train_defect_nf_count = 0
        train_good_nf_count = 0
        nf_losses = []
        for i in range(I):
            # warm-up learning rate
            lr = warmup_learning_rate(c, epoch, i+sub_epoch*I, I*c.sub_epochs, optimizers)
            # sample batch
            try:
                image, _, mask = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, _, mask = next(iterator)
            # encoder prediction
            image = image.to(c.device)
            mask = mask.to(c.device)
            with torch.no_grad():
                enc_out = encoder(image)

            # train CNF networks
            for l, layer in enumerate(pool_layers):
                if l in c.pool_layers:
                    l_idx = c.pool_layers.index(l)

                    e = activation[layer].detach()  # BxCxHxW, get input feature maps of a CNF network
                    
                    B, C, H, W = e.size()
                    S = H*W
                    E = B*S    

                    p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                    c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                    perm = torch.randperm(E).to(c.device)  # BHW, shuffle feature vectors
                    nf = nfs[l_idx]

                    # ground truth for feature maps
                    mask = mask.type(torch.float32)
                    m = F.interpolate(mask, size=(H, W), mode='nearest')
                    m[m>0]=-1
                    m[m!=-1]=1
                    m_r = m.reshape(B, 1, S).transpose(1,2).reshape(E, 1)  # BHWx1

                    FIB = E//N + int(E%N > 0)  # number of fiber batches
                    assert FIB > 0, 'MAKE SURE WE HAVE ENOUGH FIBERS, otherwise decrease N or batch-size!'
                    for f in range(FIB):
                        if f < (FIB-1):
                            idx = torch.arange(f*N, (f+1)*N)
                        else:
                            idx = torch.arange(f*N, E) # do not drop last

                        # train a CNF network
                        c_p = c_r[perm[idx]]  # NxP
                        e_p = e_r[perm[idx]]  # NxC
                        m_p = m_r[perm[idx]] #Nx1
                        z, log_jac_det = nf(e_p, [c_p,])

                        # get loss 
                        nf_log_prob, defect_logp, good_logp= get_logp(C, z, log_jac_det, m_p)
                        train_defect_nf_loss += defect_logp.sum() /C
                        train_defect_nf_count += torch.sum(m_p==-1)
                        train_good_nf_loss += good_logp.sum() /C
                        train_good_nf_count += torch.sum(m_p==1)
                        log_prob = nf_log_prob / C  # likelihood per dim
                        loss = -log_theta(log_prob)

                        optimizers.zero_grad()
                        loss.mean().backward()
                        optimizers.step()

                        if len(nf_losses) < len(c.pool_layers):
                            nf_losses.append(loss.mean()*len(loss))
                        else:
                            nf_losses[l_idx] += loss.mean()*len(loss)
                        train_loss += t2np(loss.mean()*len(loss))
                        train_count += len(loss)

        # show results
        mean_train_total_loss = train_loss / train_count
        mean_train_defect_nf_loss = train_defect_nf_loss / train_defect_nf_count
        mean_train_good_nf_loss = train_good_nf_loss / train_good_nf_count
        loss_str = ''
        for l in range(len(nf_losses)-1):
            loss_str += f'nf-{c.pool_layers[l]}_loss: {nf_losses[l]/train_count:.4f}, '
        loss_str += f'nf-{c.pool_layers[len(nf_losses)-1]}_loss: {nf_losses[len(nf_losses)-1]/train_count:.4f}'
        if c.verbose:
            print(f'Epoch: {epoch:02d}.{sub_epoch:03d}\ttrain_nf_loss: {mean_train_total_loss:.4f}, train_defect_nf_loss: {mean_train_defect_nf_loss:.4f}, train_good_nf_loss: {mean_train_good_nf_loss:.4f}, lr={lr:.6f}')
            print(loss_str)
        txt_file.write(f'\nEpoch: {epoch:02d}.{sub_epoch:03d}\ttrain_nf_loss: {mean_train_total_loss:.4f}, train_defect_nf_loss: {mean_train_defect_nf_loss:.4f}, train_good_nf_loss: {mean_train_good_nf_loss:.4f}, lr={lr:.6f}')
        txt_file.write(f'\n{loss_str}')


def test_meta_epoch(c, epoch, loader, model, pool_layers, fe_loss_fn, txt_file):
    # test
    if c.verbose:
        print('\nCompute loss and scores on test set:')
    txt_file.write('\nCompute loss and scores on test set:')

    encoder, decoder,  nfs = model
    encoder = encoder.eval()
    decoder = decoder.eval()
    nfs = [nf.eval() for nf in nfs]
    if c.add_fe_anomaly ==True:
        feature_maps = [list() for l in range(len(c.pool_layers)+len(c.pool_layers_dec)-1)]
    else:
        feature_maps = [list() for l in range(len(c.pool_layers))]

    P = c.condition_vec
    N = c.p_bs
    log_theta = torch.nn.LogSigmoid()
    height = list()
    width = list()
    image_list = list()
    gt_label_list = list()
    gt_mask_list = list()
    if 'cls' in c.loss_type:
        pred_list = list()
    elif c.loss_type =='reg':
        pred_list = [list(), list()]
    else:
        raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))
    test_dist = [list() for layer in c.pool_layers]

    test_defect_nf_loss = 0.0
    test_defect_nf_count = 0
    test_good_nf_loss = 0.0
    test_good_nf_count = 0
    test_loss = 0.0
    test_count = 0
    nf_losses = []

    with torch.no_grad():
        for i, (image, label, gt_mask) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # save
            image_list.extend(t2np(image))
            gt_label_list.extend(t2np(label))
            if gt_mask.size(1)>1:
                mask = (torch.sum(gt_mask,1, keepdim=True)>0).type(torch.float)
            else:
                mask = gt_mask
            gt_mask_list.extend(t2np(mask))

            # data
            image = image.to(c.device) # single scale
            mask = mask.to(c.device)

            # inference enc-dec
            enc_out = encoder(image)
            if c.skip_connection ==True:
                pix_out = decoder(enc_out)
            else:
                pix_out = decoder(enc_out[-1])

            if c.loss_type =='cls':
                pix_loss = fe_loss_fn(pix_out, torch.squeeze(mask,1))
                pix_loss = torch.unsqueeze(pix_loss,1)
                prob_map = torch.softmax(pix_out, 1)
                pred_list.extend(t2np(1-prob_map[:,0,:,:]))
            elif c.loss_type =='reg':
                pix_loss = torch.mean(fe_loss_fn(pix_out, image), 1, keepdim=True)
                pred_list[0].extend(t2np(pix_out))
                pred_list[1].extend(t2np(torch.squeeze(pix_loss,1)))
            elif 'smooth' in c.loss_type:
                pix_loss = fe_loss_fn(pix_out, mask.type(torch.float))
                pred_list.extend(t2np(torch.squeeze(pix_out,1)))
            else:
                raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))

            # save feature maps
            if c.is_train ==False:
                # encoder feature maps
                for l, layer in enumerate(pool_layers):
                    if l in c.pool_layers:
                        l_idx = c.pool_layers.index(l)
                        e = activation[layer].detach()  # bxcxhxw

                        _, c_idx = torch.topk(torch.mean(e, (-2,-1)), int(c.feat_avg_topk*e.size(1)), 1)

                        feat_map_sorted = [] 
                        for b in range(c_idx.size(0)):
                            feat_map_sorted.append(e[b,c_idx[b],:,:])
                        feat_map_sorted_mean = torch.unsqueeze(torch.mean(torch.stack(feat_map_sorted,0), 1), 1)
                        feat_map = F.interpolate(feat_map_sorted_mean, size=(image.size(-2), image.size(-1)), mode = 'bicubic', align_corners =True)
                        feature_maps[l_idx].extend(t2np(torch.squeeze(feat_map, 1)))

                if c.add_fe_anomaly ==True:
                    # decoder feature maps
                    dec_pool_layers = list(reversed(c.pool_layers)) 
                    for l, layer in enumerate(dec_pool_layers):
                        layer = f'dec_layer{layer}'
                        if l>0:
                            d = dec_activation[layer].detach()  # bxcxhxw

                            _, c_idx = torch.topk(torch.mean(d, (-2,-1)), int(c.feat_avg_topk*d.size(1)), 1)
                            feat_map_sorted = [] 
                            for b in range(c_idx.size(0)):
                                feat_map_sorted.append(d[b,c_idx[b],:,:])
                            feat_map_sorted_mean = torch.unsqueeze(torch.mean(torch.stack(feat_map_sorted,0), 1), 1)
                            feat_map = F.interpolate(feat_map_sorted_mean, size=(image.size(-2), image.size(-1)), mode = 'bicubic', align_corners =True)
                            feature_maps[l-1+len(c.pool_layers)].extend(t2np(torch.squeeze(feat_map, 1)))
                        else:
                            pass

            # inference CNFs
            for l, layer in enumerate(pool_layers):
                if l in c.pool_layers:
                    l_idx = c.pool_layers.index(l)

                    e = activation[layer].detach()  # BxCxHxW

                    B, C, H, W = e.size()

                    S = H*W
                    E = B*S    
                    #
                    if i == 0:  # get stats
                        height.append(H)
                        width.append(W)

                    # get feature vectors for inference
                    p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                    c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                    nf = nfs[l_idx]
                    FIB = E//N + int(E%N > 0)  # number of fiber batches

                    # ground truth for feature maps
                    mask = mask.type(torch.float32)
                    m = F.interpolate(mask, size=(H, W), mode='nearest')
                    m[m>0]=-1
                    m[m!=-1]=1
                    m_r = m.reshape(B, 1, S).transpose(1,2).reshape(E, 1)  # BHWx1
                    for f in range(FIB):
                        if f < (FIB-1):
                            idx = torch.arange(f*N, (f+1)*N)
                        else:
                            idx = torch.arange(f*N, E)

                        # inference of a CNF network
                        c_p = c_r[idx]  # NxP
                        e_p = e_r[idx]  # NxC
                        m_p = m_r[idx]  #Nx1
                        z, log_jac_det = nf(e_p, [c_p,])

                        # get loss
                        nf_log_prob, defect_logp, good_logp= get_logp(C, z, log_jac_det, m_p)

                        test_defect_nf_loss += defect_logp.sum() /C
                        test_defect_nf_count += torch.sum(m_p==-1)
                        test_good_nf_loss += good_logp.sum() /C
                        test_good_nf_count += torch.sum(m_p==1)
                        log_prob = nf_log_prob / C  # likelihood per dim

                        # save log-likelihood
                        test_dist[l_idx] = test_dist[l_idx] + log_prob.detach().cpu().tolist()
                        
                        loss = -log_theta(log_prob)
                        if len(nf_losses) < len(c.pool_layers):
                            nf_losses.append(loss.mean()*len(loss))
                        else:
                            nf_losses[l_idx] += loss.mean()*len(loss)
                        test_loss += t2np(loss.mean()*len(loss))
                        test_count += len(loss)
                else:
                    pass

    # show results
    mean_test_loss = test_loss / test_count
    mean_test_defect_nf_loss = test_defect_nf_loss / test_defect_nf_count
    mean_test_good_nf_loss = test_good_nf_loss / test_good_nf_count
    loss_str = ''
    for l in range(len(nf_losses)-1):
        loss_str += f'nf-{c.pool_layers[l]}_loss: {nf_losses[l]/test_count:.4f}, '
    loss_str += f'nf-{c.pool_layers[len(nf_losses)-1]}_loss: {nf_losses[len(nf_losses)-1]/test_count:.4f}'

    if c.verbose:
        print(f'Epoch: {epoch:02d}\ttest_nf_loss: {mean_test_loss:.4f}, test_defect_nf_loss: {mean_test_defect_nf_loss:.4f}, test_good_nf_loss: {mean_test_good_nf_loss:.4f}')
        print(loss_str)
    txt_file.write(f'\nEpoch: {epoch:02d}\ttest_nf_loss: {mean_test_loss:.4f}, test_defect_nf_loss: {mean_test_defect_nf_loss:.4f}, test_good_nf_loss: {mean_test_good_nf_loss:.4f}')
    txt_file.write(f'\n{loss_str}')
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list, feature_maps, pred_list


def test_meta_fps(c, epoch, loader, model, pool_layers, fe_loss_fn, txt_file, result_dict):
    if c.verbose:
        print('\nCompute inference speed (fps) on test set:')
    txt_file.write('\nCompute inference speed (fps) on test set:')

    encoder, decoder,  nfs = model
    encoder = encoder.eval()
    if c.add_fe_anomaly ==True:
        decoder = decoder.eval()
    else:
        del decoder
    nfs = [nf.eval() for nf in nfs]

    P = c.condition_vec
    N = c.p_bs

    starter, ender = time_measure()
    with torch.no_grad():
        # warm-up
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            image = image.to(c.device) 
            enc_out = encoder(image)  
            if c.add_fe_anomaly==True:
                if c.skip_connection ==True:
                    pix_out = decoder(enc_out)
                else:
                    pix_out = decoder(enc_out[-1])
            else:
                pass

        starter.record()
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            # data
            image = image.to(c.device) # single scale

            # enc-dec inference
            enc_out = encoder(image)

            if c.add_fe_anomaly==True:
                if c.skip_connection ==True:
                    pix_out = decoder(enc_out)
                else:
                    pix_out = decoder(enc_out[-1])
            else:
                pass

            # CNF inference
            for l, layer in enumerate(pool_layers):
                if l in c.pool_layers:
                    l_idx = c.pool_layers.index(l)

                    e = activation[layer]  # BxCxHxW

                    B, C, H, W = e.size()

                    S = H*W
                    E = B*S    

                    #
                    p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                    c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                    nf = nfs[l_idx]
                    FIB = E//N + int(E%N > 0)  # number of fiber batches
                    for f in range(FIB):
                        if f < (FIB-1):
                            idx = torch.arange(f*N, (f+1)*N)
                        else:
                            idx = torch.arange(f*N, E)

                        c_p = c_r[idx]  # NxP
                        e_p = e_r[idx]  # NxC
                        z, log_jac_det = nf(e_p, [c_p,])
                else:
                    pass
        ender.record()
        torch.cuda.synchronize()
        speed_result = starter.elapsed_time(ender)

    # show results
    fps = len(loader.dataset) / (speed_result/1000)
    if c.verbose:
        print(f'Batch size: {c.batch_size}, Inference time: {speed_result:0.2f}msec,  Data num: {len(loader.dataset)}, fps: {fps:.2f} fps')
    txt_file.write(f'\nBatch size: {c.batch_size}, Inference time: {speed_result:0.2f}msec,  Data num: {len(loader.dataset)}, fps: {fps:.2f} fps')
    result_dict['batch_size'] = c.batch_size
    result_dict['inference_time (msec)'] = speed_result
    result_dict['data_num'] = len(loader.dataset)
    result_dict['fps'] = fps


def run(c):
    if c.is_train ==True:
        # log
        log_txt_path = os.path.join(c.model_dir, f'{c.train_type}_train_progress.txt')
        model_file = open(os.path.join(c.model_dir, f'{c.train_type}_model_summary.txt'), 'w')
    else: 
        # make a path of directory to save test results
        dir_feature=[]
        if c.test_data_type == 'aug':
            dir_feature.append(f'aug')
        if c.th_manual>0:
            dir_feature.append(f'manual_th_{c.th_manual:0.2f}')
        if c.is_open==True:
            dir_feature.append('imopen')
        if c.is_close==True:
            dir_feature.append('imclose')

        if c.add_fe_anomaly ==True:
            if c.get_best_w_fe==True:
                map_type = f'{c.infer_type}-find_best_w_fe'
            else:
                map_type = f'{c.infer_type}-{c.w_fe:0.2f}'
        else:
            map_type = c.infer_type

        if len(dir_feature)>0:
            str_dir_features = '-'.join(dir_feature)
            tag = os.path.join(map_type, str_dir_features)
        else:
            tag = os.path.join(map_type, 'classic')
        save_dir = os.path.join(c.model_dir, c.class_name, c.infer_type, tag)
        makedirs(save_dir)
        log_txt_path = os.path.join(save_dir, f'{map_type}_test_results.txt')
    txt_file = open(log_txt_path, 'a')


    #============================================
    #               Set Model
    #============================================

    # set the finetuned encoder
    encoder, pool_layers_total, pool_dims_total = load_encoder_arch(c)
    if c.is_train ==True:
        sample = torch.zeros(tuple([2]+c.img_dims))
        _ = encoder(sample)
        model_file.write('Encoder Summary \n')
        model_file.write(str(encoder))
    else:
        pass
    encoder = encoder.to(c.device)

    # set decoder
    if c.is_train ==True:
        c.dec_dims_fe = []
        c.pool_layers_dec = []
        if c.set_dec_dims_nf ==True:
            for l in c.pool_layers:
                layer = pool_layers_total[l]
                c.pool_layers_dec.append(layer)
                e = activation[layer].detach()  # BxCxHxW
                input_dims = [pool_dims_total[l]]+[e.size(-2), e.size(-1)]
                c.dec_dims_fe.append(input_dims)
        else:
            for l in range(len(pool_layers_total)):
                layer = pool_layers_total[l]
                c.pool_layers_dec.append(layer)
                e = activation[layer].detach()  # BxCxHxW
                input_dims = [pool_dims_total[l]]+[e.size(-2), e.size(-1)]
                c.dec_dims_fe.append(input_dims)
    else:
        pass


    decoder, dec_pool_layers, dec_pool_dims= load_decoder_arch(c, c.dec_dims_fe) 
    print(dec_pool_layers)
    if c.is_train ==True:
        model_file.write('Decoder Summary \n')
        if c.skip_connection ==True:
            model_file.write(str(decoder))
        else:
            dec_stat=summary(decoder,c.dec_dims_fe[-1], depth=5)
            model_file.write(str(dec_stat))
    else:
        pass
    decoder = decoder.to(c.device)

    model_loaded = [encoder, decoder] 
    weight_dir = os.path.join(c.model_dir, 'weights')
    load_weights(c, model_loaded, 'fe_only') # load finetuned weights

    # set CNF networks
    L = len(c.pool_layers) 
    print('Number of pool layers =', L)

    nfs = []
    if c.is_train ==True:
        c.dec_dims = []
        for l in c.pool_layers:
            layer = pool_layers_total[l]
            e = activation[layer].detach()  # BxCxHxW
            input_dims = [pool_dims_total[l]]+[e.size(-2), e.size(-1)]
            c.dec_dims.append(input_dims)
            nfs += [load_nf_arch(c, input_dims[0])]
    else:
        for l in range(len(c.pool_layers)):
            nfs += [load_nf_arch(c, c.dec_dims[l][0])]
    nfs = [nf.to(c.device) for nf in nfs]
    print('dec_dims')
    print(c.dec_dims)

    nf_params = list(nfs[0].parameters())
    if c.is_train ==True:
        model_file.write('\n\nNF(1) Structure \n')
        model_file.write(str(nfs[0]))
    for l in range(1,len(nfs)):
        nf_params += list(nfs[l].parameters())
        if c.is_train ==True:
            model_file.write(f'\n\nNF({l+1}) Structure \n')
            model_file.write(str(nfs[l]))

    if c.is_train ==True:
        model_file.close()

    model = [encoder, decoder, nfs]

    optimizers = torch.optim.Adam([ {'params': nf_params}], lr=c.lr)

    #============================================
    #               Set Data
    #============================================
    # data
    kwargs = {'num_workers': c.workers, 'pin_memory': True} if c.use_cuda else {}

    # task data
    if c.dataset == 'mvtec' or c.dataset == 'btad':
        train_dataset = CustomDataset(c.img_size, c.data_path, c.class_name, c.dataset, c.norm_mean, c.norm_std, is_train=True)
        data_aug_path = make_aug_dataset_path(c, c.data_aug_path)
        test_aug_dataset = CustomDataset(c.img_size, data_aug_path, c.class_name, c.dataset, c.norm_mean, c.norm_std, is_train=False)
        test_dataset = CustomDataset(c.img_size, c.data_path, c.class_name, c.dataset, c.norm_mean, c.norm_std, is_train=False)
    else:
        raise NotImplementedError('{} is not supported dataset!'.format(c.dataset))

    #
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=c.batch_size, shuffle=True, drop_last=c.drop_last, **kwargs)
    test_aug_loader = torch.utils.data.DataLoader(test_aug_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=c.batch_size, shuffle=False, drop_last=False, **kwargs)
    print('train/test_aug/test loader length', len(train_loader.dataset), len(test_aug_loader.dataset), len(test_loader.dataset))
    print('train/test_aug/test loader batches', len(train_loader), len(test_aug_loader), len(test_loader))
    txt_file.write(f'\ntrain/test_aug/test loader length {len(train_loader.dataset)}, {len(test_aug_loader.dataset)}, {len(test_loader.dataset)}')
    txt_file.write(f'\ntrain/test_aug/test loader batches {len(train_loader)}, {len(test_aug_loader)}, {len(test_loader)}')

    #============================================
    #               Set Metrics
    #============================================
    # stats
    result_dict = dict()
    det_roc_obs = Score_Observer('DET_AUROC')
    seg_roc_obs = Score_Observer('PIX_AUROC')
    seg_pr_obs = Score_Observer('PIX_AUPR')
    aug_det_roc_obs = Score_Observer('DET_AUROC (AUG)')
    aug_seg_roc_obs = Score_Observer('PIX_AUROC (AUG)')
    aug_seg_pr_obs = Score_Observer('PIX_AUPR (AUG)')
    # set loss function for enc-dec network
    if c.loss_type == 'cls':
        if c.w_defect>0:
            class_weights = torch.FloatTensor([1-c.w_defect, c.w_defect]).to(c.device)
        else:
            class_weights = torch.FloatTensor([0.5, 0.5]).to(c.device)
        fe_loss_fn = nn.CrossEntropyLoss(weight = class_weights, reduction='none') 
    elif c.loss_type == 'reg': 
        fe_loss_fn = nn.L1Loss(reduction='none') 
    elif 'smooth' in c.loss_type:
        fe_loss_fn = nn.BCELoss(reduction='none') 
    else:
        raise NotImplementedError('{} is not supported loss_type!'.format(c.loss_type))

    #============================================
    #               Train or Test
    #============================================
    for epoch in range(c.meta_epochs):
        if c.is_train ==True:
            print('Train meta epoch: {}'.format(epoch))
            txt_file.write('\n\nTrain meta epoch: {}'.format(epoch))
            train_meta_epoch(c, epoch, train_loader, model, optimizers, pool_layers_total, fe_loss_fn, txt_file)
            height, width, test_aug_image_list, test_aug_dist, gt_aug_label_list, gt_aug_mask_list, feature_map_list, pred_aug_list= test_meta_epoch(c, epoch, test_aug_loader, model, pool_layers_total, fe_loss_fn, txt_file)
            height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, feature_map_list, pred_list= test_meta_epoch(c, epoch, test_loader, model, pool_layers_total, fe_loss_fn, txt_file)
            _, _, _, train_dist, _, train_gt_mask_list, _, train_pred_list= test_meta_epoch(c, epoch, train_loader, model, pool_layers_total, fe_loss_fn, txt_file)
        else:
            if 'joint' in c.infer_type:
                cal_model_size_MB(model, result_dict)
            else:
                cal_model_size_MB([encoder, nfs], result_dict)
            state = load_weights(c, model, c.infer_type)
            c_loaded = state['args']
            epoch = c_loaded.meta_epochs 
            if c.test_data_type == 'aug':
                test_meta_fps(c, epoch, test_aug_loader, model, pool_layers_total, fe_loss_fn, txt_file, result_dict)
                _, _, _, train_dist, _, train_gt_mask_list, _, train_pred_list= test_meta_epoch(c, epoch, train_loader, model, pool_layers_total, fe_loss_fn, txt_file)
                height, width, test_aug_image_list, test_aug_dist, gt_aug_label_list, gt_aug_mask_list, feature_map_list, pred_aug_list= test_meta_epoch(c, epoch, test_aug_loader, model, pool_layers_total, fe_loss_fn, txt_file)
            else:
                test_meta_fps(c, epoch, test_loader, model, pool_layers_total, fe_loss_fn, txt_file, result_dict)
                _, _, _, train_dist, _, train_gt_mask_list, _, train_pred_list= test_meta_epoch(c, epoch, train_loader, model, pool_layers_total, fe_loss_fn, txt_file)
                height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, feature_map_list, pred_list= test_meta_epoch(c, epoch, test_loader, model, pool_layers_total, fe_loss_fn, txt_file)

        # get test results and export visulaizations
        if c.is_train ==True:
            anomaly_cal = Anomaly_Score_Calculator(c.pool_layers, c.crp_size, height, width, train_pred_list, train_dist, train_gt_mask_list, c.train_type, c.pro) 
            anomaly_cal.save_results(epoch, test_aug_dist, pred_aug_list, gt_aug_mask_list, gt_aug_label_list, aug_det_roc_obs, aug_seg_roc_obs, aug_seg_pr_obs, txt_file, result_dict)
            anomaly_cal.save_results(epoch, test_dist, pred_list, gt_mask_list, gt_label_list, det_roc_obs, seg_roc_obs, seg_pr_obs, txt_file, result_dict)
            save_weights(c, model, c.model_dir) 
        else:
            anomaly_cal = Anomaly_Score_Calculator(c.pool_layers, c.crp_size, height, width, train_pred_list, train_dist, train_gt_mask_list, c.infer_type, c.pro, c.w_fe, c.get_best_w_fe) 
            if c.test_data_type == 'aug':
                anomaly_cal.save_results(epoch, test_aug_dist, pred_aug_list, gt_aug_mask_list, gt_aug_label_list, aug_det_roc_obs, aug_seg_roc_obs, aug_seg_pr_obs, txt_file, result_dict)
                gt_mask = anomaly_cal.gt_mask
                super_mask = anomaly_cal.super_mask
                gt_label = anomaly_cal.gt_label
                score_label = anomaly_cal.score_label
                w_fe = anomaly_cal.w_fe
                if c.viz:
                    viz(c, test_aug_loader, gt_label, score_label, test_aug_image_list, super_mask, gt_mask, pred_aug_list, feature_map_aug_list, txt_file, save_dir, result_dict, w_fe)
            else:
                anomaly_cal.save_results(epoch, test_dist, pred_list, gt_mask_list, gt_label_list, det_roc_obs, seg_roc_obs, seg_pr_obs, txt_file, result_dict)
                gt_mask = anomaly_cal.gt_mask
                super_mask = anomaly_cal.super_mask
                gt_label = anomaly_cal.gt_label
                score_label = anomaly_cal.score_label
                w_fe = anomaly_cal.w_fe
                if c.viz:
                    viz(c, test_loader, gt_label, score_label, test_image_list, super_mask, gt_mask, pred_list, feature_map_list, txt_file, save_dir, result_dict, w_fe)
            if c.viz:
                plot_pix_anomaly_histogram(c, save_dir, anomaly_cal, 'normal')
                if 'joint' in c.infer_type:
                    plot_pix_anomaly_histogram(c, save_dir, anomaly_cal, 'score_aggregation') # compare histograms of the combined prediction and the prediction of CNF networks
                else:
                    pass
            break

    # save test results for the last epoch
    if c.is_train ==True:
        save_results(aug_det_roc_obs, aug_seg_roc_obs, aug_seg_pr_obs, c.model_dir, c.class_name, c.run_date, 'aug')
        save_results(det_roc_obs, seg_roc_obs, seg_pr_obs, c.model_dir, c.class_name, c.run_date)
    return log_txt_path

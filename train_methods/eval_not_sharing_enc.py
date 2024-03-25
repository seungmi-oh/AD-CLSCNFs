import os, time, gc
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


def test_meta_epoch(c, epoch, loader, model, pool_layers, fe_loss_fn, txt_file, base_model, eval_not_shared = True):
    # show results only when evaluate CLS-CNFs-ns (the combined model not sharing feature extractors)
    if eval_not_shared ==True:
        if c.verbose:
            print('\nCompute loss and scores on test set:')
        txt_file.write('\nCompute loss and scores on test set:')

    base_encoder, base_nfs = base_model # CFlow-AD
    encoder, decoder,  nfs = model # the proposed method
    base_encoder = base_encoder.eval()
    base_nfs = [nf.eval() for nf in base_nfs]
    encoder = encoder.eval()
    decoder = decoder.eval()
    nfs = [nf.eval() for nf in nfs]
    feature_maps = [list() for l in range(len(c.pool_layers)+len(c.pool_layers_dec)-1)]

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
            image = image.to(c.device) 
            mask = mask.to(c.device)

            enc_out = encoder(image) # get feature maps of finetune encoder
            base_enc_out = base_encoder(image) # get feature maps of pretrained encoder

            # get decoder prediction
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

            # save feature maps of the pretrained feature extractor
            for l, layer in enumerate(pool_layers):
                if l in c.pool_layers:
                    l_idx = c.pool_layers.index(l)
                    e = base_enc_out[l].detach()  

                    _, c_idx = torch.topk(torch.mean(e, (-2,-1)), int(c.feat_avg_topk*e.size(1)), 1)

                    feat_map_sorted = [] 
                    for b in range(c_idx.size(0)):
                        feat_map_sorted.append(e[b,c_idx[b],:,:])
                    feat_map_sorted_mean = torch.unsqueeze(torch.mean(torch.stack(feat_map_sorted,0), 1), 1)
                    feat_map = F.interpolate(feat_map_sorted_mean, size=(image.size(-2), image.size(-1)), mode = 'bicubic', align_corners =True)
                    feature_maps[l_idx].extend(t2np(torch.squeeze(feat_map, 1)))

            # get feature maps of the decoder
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

            # inference
            for l, layer in enumerate(pool_layers):
                if l in c.pool_layers:
                    l_idx = c.pool_layers.index(l)

                    if eval_not_shared ==True:
                        e = base_enc_out[l].detach()  # BxCxHxW
                    else:
                        e = enc_out[l].detach()  # BxCxHxW

                    B, C, H, W = e.size()
                    S = H*W
                    E = B*S    

                    if i == 0:  # get stats
                        height.append(H)
                        width.append(W)

                    p = positionalencoding2d(P, H, W).to(c.device).unsqueeze(0).repeat(B, 1, 1, 1)
                    c_r = p.reshape(B, P, S).transpose(1, 2).reshape(E, P)  # BHWxP
                    e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
                    if eval_not_shared ==True:
                        nf = base_nfs[l_idx]
                    else:
                        nf = nfs[l_idx]

                    mask = mask.type(torch.float32)
                    m = F.interpolate(mask, size=(H, W), mode='nearest')
                    m[m>0]=-1
                    m[m!=-1]=1
                    m_r = m.reshape(B, 1, S).transpose(1,2).reshape(E, 1)  # BHWx1

                    FIB = E//N + int(E%N > 0)  # number of fiber batches
                    for f in range(FIB):
                        if f < (FIB-1):
                            idx = torch.arange(f*N, (f+1)*N)
                        else:
                            idx = torch.arange(f*N, E)

                        c_p = c_r[idx]  # NxP
                        e_p = e_r[idx]  # NxC
                        m_p = m_r[idx]  #Nx1
                        z, log_jac_det = nf(e_p, [c_p,])

                        nf_log_prob, defect_logp, good_logp= get_logp(C, z, log_jac_det, m_p)

                        test_defect_nf_loss += defect_logp.sum() /C
                        test_defect_nf_count += torch.sum(m_p==-1)
                        test_good_nf_loss += good_logp.sum() /C
                        test_good_nf_count += torch.sum(m_p==1)
                        log_prob = nf_log_prob / C  # likelihood per dim
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

    mean_test_loss = test_loss / test_count
    mean_test_defect_nf_loss = test_defect_nf_loss / test_defect_nf_count
    mean_test_good_nf_loss = test_good_nf_loss / test_good_nf_count

    # show results only when evaluate CLS-CNFs-ns (the combined model not sharing feature extractors)
    if eval_not_shared ==True:
        loss_str = ''
        for l in range(len(nf_losses)-1):
            loss_str += f'nf-{c.pool_layers[l]}_loss: {nf_losses[l]/test_count:.4f}, '
        loss_str += f'nf-{c.pool_layers[len(nf_losses)-1]}_loss: {nf_losses[len(nf_losses)-1]/test_count:.4f}'

        if c.verbose:
            print(f'Epoch: {epoch:02d}\ttest_nf_loss: {mean_test_loss:.4f}, test_defect_nf_loss: {mean_test_defect_nf_loss:.4f}, test_good_nf_loss: {mean_test_good_nf_loss:.4f}')
            print(loss_str)
        txt_file.write(f'\nEpoch: {epoch:02d}\ttest_nf_loss: {mean_test_loss:.4f}, test_defect_nf_loss: {mean_test_defect_nf_loss:.4f}, test_good_nf_loss: {mean_test_good_nf_loss:.4f}')
        txt_file.write(f'\n{loss_str}')
    else:
        pass
    return height, width, image_list, test_dist, gt_label_list, gt_mask_list, feature_maps, pred_list


def test_meta_fps(c, epoch, loader, model, pool_layers, fe_loss_fn, txt_file, base_model, result_dict):
    # test
    if c.verbose:
        print('\nCompute inference speed (fps) on test set:')
    txt_file.write('\nCompute inference speed (fps) on test set:')

    base_encoder, base_nfs = base_model
    encoder, decoder,  nfs = model
    base_encoder = base_encoder.eval()
    base_nfs = [nf.eval() for nf in base_nfs]
    encoder = encoder.eval()
    decoder = decoder.eval()
    nfs = [nf.eval() for nf in nfs]

    P = c.condition_vec
    N = c.p_bs

    starter, ender = time_measure()
    with torch.no_grad():
        # warm-up
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            image = image.to(c.device) 
            enc_out = encoder(image)  
            _= base_encoder(image)  
            if c.add_fe_anomaly==True:
                if c.skip_connection ==True:
                    pix_out = decoder(enc_out)
                else:
                    pix_out = decoder(enc_out[-1])
            else:
                pass

        # measure inference time
        starter.record()
        for i, (image, _, _) in enumerate(tqdm(loader, disable=c.hide_tqdm_bar)):
            image = image.to(c.device) 
            enc_out = encoder(image)
            base_enc_out = base_encoder(image)

            if c.skip_connection ==True:
                pix_out = decoder(enc_out)
            else:
                pix_out = decoder(enc_out[-1])

            for l, layer in enumerate(pool_layers):
                if l in c.pool_layers:
                    l_idx = c.pool_layers.index(l)

                    e = base_enc_out[l].detach()  # BxCxHxW

                    B, C, H, W = e.size()
                    S = H*W
                    E = B*S    

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


def test(c):
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

    # set the finetuned feature extractor
    encoder, pool_layers_total, pool_dims_total = load_encoder_arch(c)
    if c.is_train==True:
        sample = torch.zeros((2,3,256,256))
        _ = encoder(sample)
        model_file.write('Encoder Summary \n')
        model_file.write(str(encoder))
    else:
        pass
    encoder = encoder.to(c.device)

    # set decoder
    decoder, dec_pool_layers, dec_pool_dims= load_decoder_arch(c, c.dec_dims_fe) 
    decoder = decoder.to(c.device)

    model_loaded = [encoder, decoder] 
    weight_dir = os.path.join(c.model_dir, 'weights')
    load_weights(c, model_loaded, 'fe_only') # load enc-dec weights

    # set CNF networks
    L = len(c.pool_layers) 
    print('Number of pool layers =', L)

    nfs = []
    for l in range(len(c.pool_layers)):
        nfs += [load_nf_arch(c, c.dec_dims[l][0])]
    nfs = [nf.to(c.device) for nf in nfs]
    print('dec_dims')
    print(c.dec_dims)

    model = [encoder, decoder, nfs]

    # set the pretrained feature extractor
    base_encoder, pool_layers_total, pool_dims_total = load_encoder_arch(c)
    base_encoder = base_encoder.to(c.device)

    # set CNF networks with the pretrianed feature extractor
    L = len(c.pool_layers) 
    print('Number of pool layers =', L)

    base_nfs = []
    for l in range(len(c.pool_layers)):
        base_nfs += [load_nf_arch(c, c.dec_dims[l][0])]
    base_nfs = [nf.to(c.device) for nf in base_nfs]

    base_model = [base_encoder, base_nfs]


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
    txt_file.write(f'\ntrain/test loader length {len(train_loader.dataset)}, {len(test_aug_loader.dataset)}, {len(test_loader.dataset)}')
    txt_file.write(f'\ntrain/test loader batches {len(train_loader)}, {len(test_aug_loader)}, {len(test_loader)}')

    #============================================
    #               Set Metrics
    #============================================
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
    result_dict = dict()
    cal_model_size_MB(model_loaded+base_model, result_dict)
    state = load_weights(c, model, c.infer_type)
    _ = load_weights(c, base_model, 'base-nf_only')
    c_loaded = state['args']
    epoch = c_loaded.meta_epochs 

    # not shared encoder
    test_meta_fps(c, epoch, test_loader, model, pool_layers_total, fe_loss_fn, txt_file, base_model, result_dict)
    height, width, test_image_list, base_test_dist, gt_label_list, gt_mask_list, base_feature_map_list, pred_list= test_meta_epoch(c, epoch, test_loader, model, pool_layers_total, fe_loss_fn, txt_file, base_model)
    _, _, _, base_train_dist, _, train_gt_mask_list, _, train_pred_list= test_meta_epoch(c, epoch, train_loader, model, pool_layers_total, fe_loss_fn, txt_file, base_model)

    # shared encoder
    height, width, test_image_list, test_dist, gt_label_list, gt_mask_list, feature_map_list, pred_list= test_meta_epoch(c, epoch, test_loader, model, pool_layers_total, fe_loss_fn, txt_file, base_model, False)
    _, _, _, train_dist, _, train_gt_mask_list, _, train_pred_list= test_meta_epoch(c, epoch, train_loader, model, pool_layers_total, fe_loss_fn, txt_file, base_model, False)

    # compare the performance of CNF networks with the pretrained feature extractor and finetuned feature extractor (nf_only) 
    # compare the performance of combined networks sharing feature extractors and not sharing feature extractors (c.infer_type --> joint_not_sharing) 
    for infer_type in ['nf_only', c.infer_type]:
        det_roc_obs = Score_Observer('DET_AUROC')
        seg_roc_obs = Score_Observer('PIX_AUROC')
        seg_pr_obs = Score_Observer('PIX_AUPR')
        base_det_roc_obs = Score_Observer('DET_AUROC (Base)')
        base_seg_roc_obs = Score_Observer('PIX_AUROC (Base)')
        base_seg_pr_obs = Score_Observer('PIX_AUPR (Base)')
        base_anomaly_cal = Anomaly_Score_Calculator(c.pool_layers, c.crp_size, height, width, train_pred_list, base_train_dist, train_gt_mask_list, infer_type, c.pro, c.w_fe, c.get_best_w_fe) 
        anomaly_cal = Anomaly_Score_Calculator(c.pool_layers, c.crp_size, height, width, train_pred_list, train_dist, train_gt_mask_list, infer_type, c.pro, c.w_fe, c.get_best_w_fe) 
        if 'joint' in infer_type:
            base_anomaly_cal.save_results(epoch, base_test_dist, pred_list, gt_mask_list, gt_label_list, base_det_roc_obs, base_seg_roc_obs, base_seg_pr_obs, txt_file, result_dict)
        else:
            base_anomaly_cal.save_results(epoch, base_test_dist, pred_list, gt_mask_list, gt_label_list, base_det_roc_obs, base_seg_roc_obs, base_seg_pr_obs, txt_file, result_dict, False)
        anomaly_cal.save_results(epoch, test_dist, pred_list, gt_mask_list, gt_label_list, det_roc_obs, seg_roc_obs, seg_pr_obs, txt_file, result_dict, False)
        gt_mask = anomaly_cal.gt_mask
        base_super_mask = base_anomaly_cal.super_mask
        super_mask = anomaly_cal.super_mask
        gt_label = anomaly_cal.gt_label
        score_label = base_anomaly_cal.score_label
        w_fe = anomaly_cal.w_fe
        if c.viz:
            if 'joint' in infer_type:
                viz(c, test_loader, gt_label, score_label, test_image_list, base_super_mask, gt_mask, pred_list, base_feature_map_list, txt_file, save_dir, result_dict, w_fe)
                plot_pix_anomaly_histogram(c, save_dir, base_anomaly_cal, 'normal')
                plot_pix_anomaly_histogram(c, save_dir, anomaly_cal, 'finetune_encoder', base_anomaly_cal, True)
            else:
                plot_pix_anomaly_histogram(c, save_dir, anomaly_cal, 'finetune_encoder', base_anomaly_cal, False)
    return log_txt_path

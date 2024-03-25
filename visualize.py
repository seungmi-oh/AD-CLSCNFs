'''This code is based on the CFlow-AD project (source: https://github.com/gudovskiy/cflow-ad/tree/master).
We modified and added the necessary modules or functions for our purposes.'''
import os
import datetime
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, roc_curve, f1_score, auc
from skimage import morphology
from skimage.segmentation import mark_boundaries
import torchvision as tv
import matplotlib.pyplot as plt
from model import activation
import matplotlib
matplotlib.use('Agg')
from utils import *

norm = matplotlib.colors.Normalize(vmin=0.0, vmax=255.0)
cm = 1/2.54
dpi = 300

def denormalization(x, norm_mean, norm_std):
    mean = np.array(norm_mean)
    std = np.array(norm_std)
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


# plot histogram
def export_hist(c, gts, scores, detect_type, save_dir):
    image_dirs = os.path.join(save_dir, 'stats')
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    makedirs(image_dirs)
    Y = scores.flatten()
    Y_label = gts.flatten()
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    defect_num = np.sum(Y_label==1)
    plt.hist([Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['r', 'g'], label=['ANO', 'TYP'], alpha=0.3, histtype='stepfilled')
    plt.legend()
    image_file = os.path.join(image_dirs, f'hist_plots_'+ detect_type + '_' + c.run_date.replace(':','')+'.png')
    fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()


# save visualization results
def export_test_images(c, test_img, gts, scores, pix_imgs, feat_maps, threshold, file_list, save_dir, txt_file, w_fe, result_dict):
    # set directory
    if c.add_fe_anomaly==True:
        image_dirs = os.path.join(save_dir, f'viz-{w_fe:0.2f}')
    else:
        image_dirs = os.path.join(save_dir, f'viz')

    print('Exporting images...')
    num = len(test_img)
    if c.is_k_disk==True:
        kernel = morphology.disk(c.k_size)
    else:
        kernel = np.ones(c.k_size)
    if c.infer_type == 'fe_only':
        scores_norm = 1.0
    else:
        scores_norm = 1.0/scores.max()
    seg_performance = [0,0,0] 
    jet_8 = plt.get_cmap('jet', lut=2**8)

    for i in range(num):
        # image
        img = test_img[i]
        img = denormalization(img, c.norm_mean, c.norm_std)

        if len(pix_imgs)==len(test_img):
            preds = pix_imgs[i]
            pred_img = (255.0*preds).astype(np.uint8)
            preds_img_8 = jet_8(pred_img)
            preds_img = [np.uint8(preds_img_8[:,:,:3]*255)]
        elif len(pix_imgs)==2: # for pixel-wise regression
            preds = pix_imgs[1][i]
            pred_img = (255.0*preds).astype(np.uint8)
            preds_img_8 = jet_8(pred_img)
            preds_img_ = np.uint8(preds_img_8[:,:,:3]*255)
            recons = denormalization(pix_imgs[0][i], c.norm_mean, c.norm_std)
            recons_bgr = cv2.cvtColor(recons, cv2.COLOR_RGB2BGR)
            preds_img = [recons_bgr, preds_img_]
        else: 
            preds_img = []

       # gts
        gt_mask = gts[i].astype(np.float64)
        gt_mask = (255.0*gt_mask).astype(np.uint8)
        dilate_kernel = np.ones((3,3), dtype=np.uint8)
        gt_mask = cv2.dilate(gt_mask, dilate_kernel, iterations=1)

        # anomaly prediction
        score_mask = np.zeros_like(scores[i])
        score_mask[scores[i] >=  threshold] = 1.0
        score_mask = (255.0*score_mask).astype(np.uint8)
        if c.is_close == True and c.is_k_disk==True:
            score_mask = morphology.closing(score_mask, kernel)
        elif c.is_open == True and c.is_k_disk ==True:
            score_mask = morphology.opening(score_mask, kernel)
        elif c.is_open==False and c.is_close ==False:
            pass
        else:
            print('Morphology option is wrong!')
            raise KeyboardInterrupt
        score_mask = (255.0*score_mask).astype(np.uint8)

        # nf scores
        if len(preds_img)>0 and c.infer_type != 'fe_only':
            # aggregated scores
            if c.add_fe_anomaly==True:
                score_nf = scores[i]-w_fe*preds
                score_nf_norm = 1.0/score_nf.max()
                score_map = (255.0*score_nf*score_nf_norm).astype(np.uint8)
                score_map_8 = jet_8(score_map)
                score_map = score_map_8[:,:,:3]*255

                scores_tot_norm = 1.0/scores.max()
                score_map_tot = (255.0*scores[i]*scores_tot_norm).astype(np.uint8)
                score_map_tot_8 = jet_8(score_map_tot)
                score_map_tot = score_map_tot_8[:,:,:3]*255
                score_map_list = [score_map, score_map_tot]
            # nf score only
            else:
                score_nf = scores[i]
                score_nf_norm = 1.0/score_nf.max()
                score_map = (255.0*score_nf*score_nf_norm).astype(np.uint8)
                score_map_8 = jet_8(score_map)
                score_map = score_map_8[:,:,:3]*255
                score_map_list = [score_map]
        # fe score
        else:
            score_map = (255.0*scores[i]*scores_norm).astype(np.uint8)
            score_map_8 = jet_8(score_map)
            score_map = score_map_8[:,:,:3]*255
            score_map_list = [score_map]

        # ground truth 
        gt_3d = np.transpose(np.array([gt_mask]*3), (1,2,0))

        # feature maps
        gray_img = np.uint8(np.transpose(np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)]*3), (1,2,0)))
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if len(feat_maps) != len(scores):
            normed_feat_maps = []
            l=0
            for feat_map_ in feat_maps: 
                l=l+1 
                mean_feat_map = feat_map_[i]
                mean_normed_feat_map = (255*rescale(mean_feat_map)).astype(np.uint8)
                mean_feat_map_8 = jet_8(mean_normed_feat_map)
                mean_normed_feat_map = mean_feat_map_8[:,:,:3]*255
                normed_feat_maps.append(mean_normed_feat_map)
        else:
            feat_map = feat_maps[i]
            if len(feat_map.shape)>2:
                feat_map = np.mean(feat_maps[i],0)
            else:
                pass
            normed_feat_map = (255*rescale(feat_map)).astype(np.uint8)
            feat_map_8 = jet_8(normed_feat_map)
            normed_feat_map = feat_map_8[:,:,:3]*255
            normed_feat_maps = [normed_feat_map]

        # get seg-wise results
        image_file_path = file_list[i]
        save_file_info_ = image_file_path.replace(os.path.join(os.path.join(c.data_path, c.class_name), 'test')+os.path.sep,'') 
        save_file_info = save_file_info_[:-4]
        save_file_info_list = save_file_info.split(os.path.sep) 
        save_file_name = '-'.join(save_file_info_list[-2:])
        seg_img_list, seg_pred_list, seg_score_list, seg_smap_list, seg_gt_list, seg_feat_list, seg_file_list, img_seg_performance = get_segment_img(img, preds_img, score_mask, score_map_list, gt_mask, normed_feat_maps, save_file_name, image_dirs, c.img_dims) 


        for e in range(len(seg_performance)):
            seg_performance[e] += img_seg_performance[e]

        # detection result 
        margin = np.ones((c.img_size[0], 10, 3))*255
        for seg_i in range(len(seg_img_list)):
            c_img = seg_img_list[seg_i] 
            c_pred = seg_pred_list[seg_i]
            c_gt_mask = seg_gt_list[seg_i] 
            c_score_mask = seg_score_list[seg_i]*255 
            c_score_map = seg_smap_list[seg_i] 
            c_feat_map = seg_feat_list[seg_i]
        
            c_score_mask_3D = np.transpose(np.array([c_score_mask]*3, dtype=np.uint8), (1,2,0))
            c_score_mask_3D = np.array(c_score_mask_3D, np.uint8)

            if 'joint' in c.infer_type:
                save_img_list_temp = [c_img, c_gt_mask, c_feat_map, c_pred, c_score_map, c_score_mask_3D]
            else:
                save_img_list_temp = [c_img, c_gt_mask, c_feat_map, c_score_map, c_score_mask_3D]

            save_img_list = [] 
            for save_img_ in save_img_list_temp:
                for col in range(int(save_img_.shape[1]//c.img_size[1])):
                    img_ = save_img_[:,col*c.img_size[1]:(col+1)*c.img_size[1],:]
                    save_img_list.extend([img_, margin])

            save_img = np.concatenate(save_img_list[:-1], axis=1)
            save_img = save_img.astype(np.uint8)

            image_file = seg_file_list[seg_i]+'.jpg'
            cv2.imwrite(image_file, save_img)

    # write segmentation result
    tp, fn, fp = seg_performance
    recall = tp/(fn+tp+1e-7)
    precision = tp/(fp+tp+1e-7)
    f1_score = 2*recall*precision/(recall+precision+1e-7)
    txt_file.write(f'\nTP num: {tp}, FN num: {fn}, FP num: {fp}')
    txt_file.write(f'\nSEG Performance) recall: {recall*100.0 :0.2f}, precision: {precision*100.0:0.2f}, f1-score: {f1_score*100.0:0.2f}')
    result_dict['threshold'] = threshold
    result_dict['TP'] = tp 
    result_dict['FN'] = fn
    result_dict['FP'] = fp
    result_dict['recall'] = recall*100.0
    result_dict['precision'] = precision*100.0
    result_dict['f1-score'] = f1_score*100.0
    csv_path = os.path.join(save_dir, 'test_result.csv') 
    write_csv(result_dict, csv_path, c.class_name)


def get_segment_img(test_img, test_preds, score_mask, score_maps_list, gt_mask, feat_map, file_name, image_dirs, img_dims):
    H=img_dims[-2]
    W=img_dims[-1]
    seg_tp = 0
    seg_fn = 0
    seg_fp = 0

    # image
    if np.max(test_img)>1:
        test_img_norm = test_img
    else:
        test_img_norm = cv2.normalize(test_img, None, 0,255, cv2.NORM_MINMAX)
    gray_test_img = cv2.cvtColor(test_img_norm, cv2.COLOR_RGB2GRAY)
    bgr_test_img = cv2.cvtColor(test_img_norm, cv2.COLOR_RGB2BGR)

    # scores
    test_smap_bgr_list = []
    for score_map in score_maps_list:
        test_smap_only = np.array(score_map, dtype = np.uint8)
        test_img_gray_3D = np.transpose(np.array([gray_test_img]*3, dtype=np.uint8), (1,2,0))
        test_smap = cv2.addWeighted(test_smap_only, 0.7, test_img_gray_3D, 0.3, 0)
        test_smap_bgr = cv2.cvtColor(test_smap, cv2.COLOR_RGB2BGR)
        test_smap_bgr_list.append(test_smap_bgr)

    # gts
    if np.max(gt_mask)>1:
        gt_norm = gt_mask
    else:
        gt_norm = cv2.normalize(gt_mask, None, 0,255, cv2.NORM_MINMAX)
    if gt_norm.shape[-1]==1 or len(gt_norm.shape)==2:
        gt_bgr = np.transpose(np.array([gt_norm]*3, dtype=np.uint8), (1,2,0))
    else:
        gt_bgr = cv2.cvtColor(gt_norm, cv2.COLOR_RGB2BGR)
    
    # feature maps
    test_fmap_bgr = []
    for i in range(len(feat_map)):
        test_fmap_only = np.array(feat_map[i], dtype = np.uint8)
        test_fmap = cv2.addWeighted(test_fmap_only, 0.7, test_img_gray_3D, 0.3, 0)
        test_fmap_bgr.append(cv2.cvtColor(test_fmap, cv2.COLOR_RGB2BGR))

    # predictions
    bgr_test_preds = []
    if len(test_preds)==0:
        pass
    else:
        test_preds_only = cv2.cvtColor(test_preds[-1], cv2.COLOR_RGB2BGR)
        test_pred_bgr = cv2.addWeighted(test_preds_only, 0.7, test_img_gray_3D, 0.3, 0)
        if len(test_preds)==2:
            bgr_test_preds = [test_preds[0], test_pred_bgr]
        elif len(test_preds)==1:
            bgr_test_preds = [test_pred_bgr]

    # set directory 
    if len(image_dirs)>2:
        makedirs(os.path.join(image_dirs, 'tp'))
        makedirs(os.path.join(image_dirs, 'fp'))
        makedirs(os.path.join(image_dirs, 'fn'))
        makedirs(os.path.join(image_dirs, 'tn'))

    # get seg results
    cropped_img_list = []
    cropped_preds_list = []
    score_mask_list = []
    score_map_list = []
    gt_mask_list = []
    feat_map_list = []
    file_list = []

    # true negative
    if np.sum(gt_mask)==0 and np.sum(score_mask)==0:
        append_segment_img(bgr_test_img, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'tn', '')


    # false negative or true positive
    defect_seg_num, defect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(gt_mask)
    for seg_idx in range(1, defect_seg_num):
        gt_mask_seg_wise = np.zeros(defect_seg_wise_label.shape)
        gt_mask_seg_wise[defect_seg_wise_label==seg_idx]=255
        if np.sum(score_mask[defect_seg_wise_label==seg_idx])>0:
            seg_tp += 1
        else: 
            seg_fn += 1

    detect_seg_num, detect_seg_wise_label, bbox_info, centroids=cv2.connectedComponentsWithStats(score_mask)
    for seg_idx in range(1, detect_seg_num):
        score_mask_seg_wise = np.zeros(detect_seg_wise_label.shape)
        score_mask_seg_wise[detect_seg_wise_label==seg_idx]=255
        if np.sum(gt_mask[detect_seg_wise_label==seg_idx])>0:
            pass
        else: 
            seg_fp += 1

    if seg_fp>0:
        append_segment_img(bgr_test_img, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'fp', f'-tp_{seg_tp}-fn_{seg_fn}-fp_{seg_fp}')
    if seg_fn>0:
        append_segment_img(bgr_test_img, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'fn', f'-tp_{seg_tp}-fn_{seg_fn}-fp_{seg_fp}')
    if seg_tp>0:
        append_segment_img(bgr_test_img, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, 'tp', f'-tp_{seg_tp}-fn_{seg_fn}-fp_{seg_fp}')
        
    return cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, [seg_tp, seg_fn, seg_fp] 


# append a visualized image 
def append_segment_img(bgr_test_img, bgr_test_preds, score_mask, test_smap_bgr_list, gt_bgr, test_fmap_bgr, cropped_img_list, cropped_preds_list, score_mask_list, score_map_list, gt_mask_list, feat_map_list, file_list, file_name, image_dirs, detect_result, tag):
    cropped_img=bgr_test_img
    if len(bgr_test_preds)==0:
        cropped_preds = None
    elif len(bgr_test_preds)==1:
        cropped_preds = bgr_test_preds[0]
    else:
        cropped_preds_ = []
        for bgr_test_pred in bgr_test_preds:
            cropped_preds_.append(bgr_test_pred)
        cropped_preds = np.concatenate(cropped_preds_, axis=1) 
    cropped_gt=gt_bgr
    cropped_score=score_mask
    cropped_smaps = []
    for test_smap_bgr in test_smap_bgr_list:
        cropped_smaps.append(test_smap_bgr)
    cropped_smap=np.concatenate(cropped_smaps, axis = 1)

    cropped_fmaps = []
    for i in range(len(test_fmap_bgr)):
        cropped_fmaps.append(test_fmap_bgr[i])
    cropped_feat_map = np.concatenate(cropped_fmaps, axis=1) 

    cropped_img_list.append(cropped_img)
    cropped_preds_list.append(cropped_preds)
    score_mask_list.append(cropped_score)
    score_map_list.append(cropped_smap)
    gt_mask_list.append(cropped_gt)
    feat_map_list.append(cropped_feat_map)
    file_list.append(os.path.join(os.path.join(image_dirs, detect_result), f'{file_name}{tag}')) 


# save pixel-wise PR curves of two models that you desire to compare
def compare_scores(c, recall_list, precision_list, annot_list, detect_type, save_dir): 
    print('Exporting Recall Precision curve...')
    image_dirs = os.path.join(os.path.join(save_dir, 'stats'))
    plt.rcParams.update({'font.size': 4})
    makedirs(image_dirs)
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    color_list = ['c', 'm', 'y', 'r', 'g', 'b']
    plt.title(f'Recall Precision Curve')
    plt.plot([1,0], ls="--")
    plt.plot([1,1], [1,0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.xlabel('RECALL')
    plt.ylabel('PRECISION')

    for i in range(len(recall_list)):
        recall = recall_list[i]
        precision = precision_list[i]
        auc_pr = auc(recall, precision)
        annot = f'{annot_list[i]} aupr: {auc_pr*100:0.2f}' 
        plt.plot(recall, precision, color = color_list[i], label = annot, linewidth = 1, alpha =0.5)
    plt.legend()
    image_file = os.path.join(image_dirs, 'PR_curve_'+ detect_type + '_' + c.run_date.replace(':','') +'.png')
    fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()


# save pixel-wise PR curve of a model
def save_scores(c, recall, precision, f1, thresholds, txt_file, detect_type, save_dir): 
    print('Exporting Recall Precision curve...')
    image_dirs = os.path.join(os.path.join(save_dir, 'stats'))
    plt.rcParams.update({'font.size': 4})
    makedirs(image_dirs)

    opt_idx = np.argmax(f1)
    optimized_th = thresholds[opt_idx]
    optimized_recall = recall[opt_idx]
    optimized_pr = precision[opt_idx]
    optimized_f1 = f1[opt_idx]
    auc_pr = auc(recall, precision)

    txt_file.write(f'\n\n{detect_type.upper()} Results')
    txt_file.write(f'\nBest F1 Score: {optimized_f1*100.0: .2f}')
    txt_file.write(f'\nAUC of PR curve: {auc_pr*100.0:0.2f}')
    txt_file.write(f'\nThreshold: {optimized_th:0.2f}, RECALL: {optimized_recall*100.0:.2f}, PRECISION: {optimized_pr*100.0:.2f}')

    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)

    plt.title(f'Recall Precision Curve')
    plt.plot(recall, precision)
    plt.scatter(optimized_recall, optimized_pr)
    plt.annotate(f'Threshold: {optimized_th:0.2f}\nRECALL: {optimized_recall*100.0:.2f}\nPRECISION: {optimized_pr*100.0:.2f}', xy=(optimized_recall, optimized_pr))
    plt.plot([1,0], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.xlabel('RECALL')
    plt.ylabel('PRECISION')
    image_file = os.path.join(image_dirs, 'PR_curve_'+ detect_type + '_' + c.run_date.replace(':','')+'.png')
    fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()
    return optimized_th

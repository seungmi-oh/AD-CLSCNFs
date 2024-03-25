import numpy as np
import torch, copy
import torch.nn.functional as F
from visualize import *
from skimage import measure
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from skimage.measure import label, regionprops

# Calculate anomaly scores
class Anomaly_Score_Calculator:
    def __init__(self, pool_layers, inp_size, height, width, train_pred_list, train_dist, train_gt_mask_list, infer_type, pro, w_fe =0, best_w_fe =False):
        self.pool_layers = pool_layers
        self.inp_size = inp_size
        self.height = height
        self.width = width
        self.train_dist = train_dist
        self.train_gt_mask_list = train_gt_mask_list
        self.pro = pro

        if len(train_pred_list)==2:
            self.train_fe_super_mask = np.array(train_pred_list[-1], dtype=np.float32)
        elif len(train_pred_list)==0:
            pass
        else:
            self.train_fe_super_mask = np.array(train_pred_list, dtype=np.float32)

        self.infer_type = infer_type
        self.best_w_fe = best_w_fe
        self.w_fe = w_fe
    
    def get_train_CNFs_anomaly_score_map(self):
        train_map = [list() for p in self.pool_layers]
        self.train_max_list =[]
        for l, p in enumerate(self.pool_layers):
            train_norm = torch.tensor(self.train_dist[l], dtype=torch.double) 
            train_max = torch.max(train_norm)
            self.train_max_list.append(train_max.item())

            train_norm-=torch.max(train_norm)  # normalize likelihoods to (-Inf:0] by subtracting the maximum value for training set
            train_prob = torch.exp(train_norm) # convert to probs in range [0:1] for training set
            train_mask = train_prob.reshape(-1, self.height[l], self.width[l])

            # upsample
            train_map[l] = F.interpolate(train_mask.unsqueeze(1),
                size=self.inp_size, mode='bilinear', align_corners=True).squeeze().numpy()

        # score aggregation
        train_score_map = np.zeros_like(train_map[0])
        for l, p in enumerate(self.pool_layers):
            train_score_map += train_map[l]
        train_score_mask = train_score_map

        # invert probs to anomaly scores
        self.train_nf_super_mask = train_score_mask.max() - train_score_mask
        self.train_gt_mask = np.squeeze(np.asarray(self.train_gt_mask_list, dtype=np.bool), axis=1)

    def get_CNFs_anomaly_score_map(self, test_dist):
        test_map = [list() for p in self.pool_layers]
        for l, p in enumerate(self.pool_layers):
            test_norm = torch.tensor(test_dist[l], dtype=torch.double)  
            test_norm -= self.train_max_list[l] 
            test_prob = torch.exp(test_norm) 

            test_mask = test_prob.reshape(-1, self.height[l], self.width[l])
            # upsample
            test_map[l] = F.interpolate(test_mask.unsqueeze(1),
                size=self.inp_size, mode='bilinear', align_corners=True).squeeze().numpy()

        # score aggregation
        score_map = np.zeros_like(test_map[0])
        for l, p in enumerate(self.pool_layers):
            score_map += test_map[l]
        score_mask = score_map

        # invert probs to anomaly scores
        nf_super_mask = score_mask.max() - score_mask
        return nf_super_mask

    def get_pix_anomaly_score_map(self, pred_list):
        if len(pred_list)==2:
            fe_super_mask = np.array(pred_list[-1], dtype=np.float32)
        else:
            fe_super_mask = np.array(pred_list, dtype=np.float32)
        return fe_super_mask

    def aggregate_anomaly_score_map(self, fe_super_mask, nf_super_mask, w_fe):
        joint_super_mask = copy.deepcopy(nf_super_mask)
        joint_super_mask += w_fe*fe_super_mask
        return joint_super_mask

    # find the best weight using grid search
    def get_best_w_fe(self, fe_super_mask, nf_super_mask, gt_mask):
        best_w_fe = 0
        best_aupr = 0
        for w in range(1,21):
            w_fe = w*0.05
            joint_super_mask = self.aggregate_anomaly_score_map(fe_super_mask, nf_super_mask, w_fe)

            nf_Y = nf_super_mask.flatten()
            Y = joint_super_mask.flatten()
            Y_label = gt_mask.flatten()

            nf_precision, nf_recall, nf_thresholds = precision_recall_curve(Y_label, nf_Y)
            precision, recall, thresholds = precision_recall_curve(Y_label, Y)
            auc_pr = auc(recall, precision)
            if auc_pr == max(auc_pr, best_aupr):
                best_w_fe = w_fe
                best_aupr = auc_pr
            else:
                pass
        return best_w_fe 

    def save_results(self, epoch, test_dist, pred_list, gt_mask_list, gt_label_list, det_roc_obs, seg_roc_obs, seg_pr_obs, txt_file, result_dict, verbose=True):
        self.gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
        if self.infer_type=='fe_only':
            self.super_mask = self.get_pix_anomaly_score_map(pred_list)
        elif 'joint' in self.infer_type:
            self.fe_super_mask = self.get_pix_anomaly_score_map(pred_list)
            self.get_train_CNFs_anomaly_score_map()
            self.nf_super_mask = self.get_CNFs_anomaly_score_map(test_dist)
            if self.best_w_fe == True:
                self.w_fe = self.get_best_w_fe(fe_super_mask, nf_super_mask, self.gt_mask)
            else:
                pass
            self.super_mask = self.aggregate_anomaly_score_map(self.fe_super_mask, self.nf_super_mask, self.w_fe)
            self.train_super_mask = self.aggregate_anomaly_score_map(self.train_fe_super_mask, self.train_nf_super_mask, self.w_fe)
        else:
            self.get_train_CNFs_anomaly_score_map()
            self.super_mask = self.get_CNFs_anomaly_score_map(test_dist)
            self.train_super_mask = self.train_nf_super_mask

        self.score_label = np.max(self.super_mask, axis=(1, 2))
        self.gt_label = np.asarray(gt_label_list, dtype=np.bool)
        
        #AUROC at the image-level
        det_roc_auc = roc_auc_score(self.gt_label, self.score_label)
        _ = det_roc_obs.update(100.0*det_roc_auc, epoch, txt_file, verbose)
        if verbose ==True:
            result_dict[det_roc_obs.name] = 100.0*det_roc_auc 

        #AUPR at the pixel-level
        pix_precision, pix_recall, pix_ths = precision_recall_curve(self.gt_mask.flatten(), self.super_mask.flatten())
        seg_pr_auc = auc(pix_recall, pix_precision) 
        save_best_seg_weights = seg_pr_obs.update(100.0*seg_pr_auc, epoch, txt_file, verbose)
        if verbose ==True:
            result_dict[seg_pr_obs.name] = 100.0*seg_pr_auc 

        #AUROC at the pixel-level
        seg_roc_auc = roc_auc_score(self.gt_mask.flatten(), self.super_mask.flatten())
        save_best_seg_weights = seg_roc_obs.update(100.0*seg_roc_auc, epoch, txt_file, verbose)
        if verbose ==True:
            result_dict[seg_roc_obs.name] = 100.0*seg_roc_auc 

        #AUPRO 
        if self.pro ==True:
            seg_pro = cal_pro_metric(self.gt_mask, self.super_mask)
            if verbose ==True:
                print(f'    AUPRO: \t max: {seg_pro*100:.2f}')
                txt_file.write(f'\nPIX_AUPRO: \t max: {seg_pro*100:.2f}')
                result_dict['AUPRO'] = 100.0*seg_pro
        return result_dict


def plot_pix_anomaly_histogram(c, save_dir, anomaly_cal, compare_mode, base_anomaly_cal=None, add_fe_anomaly=None):
    if add_fe_anomaly is None:
        add_fe_anomaly = c.add_fe_anomaly
    image_dirs = os.path.join(os.path.join(save_dir, 'stats'))
    print('Exporting histogram...')
    plt.rcParams.update({'font.size': 4})
    makedirs(image_dirs)
    fig = plt.figure(figsize=(4*cm, 4*cm), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    # save histogram plot with anomaly scores of training set
    if compare_mode == 'normal':
        Y = anomaly_cal.super_mask.flatten()
        Y_label = anomaly_cal.gt_mask.flatten()
        defect_num = np.sum(Y_label==1)
        Y_train = anomaly_cal.train_super_mask.flatten() 
        Y_train_label = anomaly_cal.train_gt_mask.flatten()
        plt.hist([Y_train, Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['b', 'r', 'g'], label=['TRAIN', 'ANO', 'TYP'], alpha=0.3, histtype='stepfilled')
        image_file = os.path.join(image_dirs, f'hist_plots_pix_with_trainset_' + c.run_date.replace(':', '')+'.png')
    # save histogram plots to compare the prediction of CNF networks and the combined prediction 
    elif compare_mode == 'score_aggregation':
        Y = anomaly_cal.super_mask.flatten()
        Y_label = anomaly_cal.gt_mask.flatten()
        defect_num = np.sum(Y_label==1)
        nf_Y = anomaly_cal.nf_super_mask.flatten()
        plt.hist([nf_Y[Y_label==1], nf_Y[Y_label==0],Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['tab:purple', 'tab:cyan', 'tab:red', 'tab:green'], label=['ANO(nf)', 'TYP(nf)', 'ANO(nf+cls)', 'TYP(nf+cls)'], alpha=0.3, histtype='stepfilled')
        comp_precision, comp_recall, _= precision_recall_curve(Y_label, nf_Y)
        precision, recall, _= precision_recall_curve(Y_label, Y)
        compare_scores(c, [comp_recall, recall], [comp_precision, precision], ['nf', 'nf+cls'], f'pix_compare_{compare_mode}', save_dir)
        image_file = os.path.join(image_dirs, f'hist_plots_pix_compare_{compare_mode}_' + c.run_date.replace(':', '')+'.png')
    # save histogram plots to compare the finetuned feature extractor and the pretrained feature extractor 
    elif compare_mode == 'finetune_encoder':
        w_fe_base = base_anomaly_cal.w_fe
        w_fe = anomaly_cal.w_fe
        Y = anomaly_cal.super_mask.flatten()
        Y_label = anomaly_cal.gt_mask.flatten()
        defect_num = np.sum(Y_label==1)
        base_Y = base_anomaly_cal.super_mask.flatten()
        plt.hist([base_Y[Y_label==1], base_Y[Y_label==0],Y[Y_label==1], Y[Y_label==0]], 500, density=True, color=['tab:purple', 'tab:cyan', 'tab:red', 'tab:green'], label=['ANO(pretrained)', 'TYP(pretrained)', 'ANO(finetuned)', 'TYP(finetuned)'], alpha=0.3, histtype='stepfilled')

        comp_precision, comp_recall, _= precision_recall_curve(Y_label, base_Y)
        precision, recall, _= precision_recall_curve(Y_label, Y)
        if add_fe_anomaly ==True:
            compare_scores(c, [comp_recall, recall], [comp_precision, precision], ['pre-trained', 'fine-tuned'], f'pix_compare_{compare_mode}-pretrain_{w_fe_base:0.2f}-finetune_{w_fe:0.2f}', save_dir)
            image_file = os.path.join(image_dirs, f'hist_plots_pix_compare_{compare_mode}_base_{w_fe_base:0.2f}_proposed_{w_fe:0.2f}_' + c.run_date.replace(':', '')+'.png')
        else:
            compare_scores(c, [comp_recall, recall], [comp_precision, precision], ['pre-trained', 'fine-tuned'], f'pix_compare_{compare_mode}', save_dir)
            image_file = os.path.join(image_dirs, f'hist_plots_pix_compare_{compare_mode}_' + c.run_date.replace(':', '')+'.png')
    else:
        NotImplementedError(f'{compare_mode} is not implemented for compare_mode')

    plt.legend(loc = 'center left', bbox_to_anchor =(1,0.5))
    fig.savefig(image_file, dpi=dpi, format='png', bbox_inches = 'tight', pad_inches = 0.0)
    plt.close()


# visualize inference results
def viz(c, test_loader, gt_label, score_label, test_img, super_mask, gt_mask, pix_image, feature_maps, txt_file, save_dir, result_dict, w_fe=0):
    test_dataset = test_loader.dataset
    det_precision, det_recall, det_thresholds = precision_recall_curve(gt_label, score_label)
    a = 2 * det_precision * det_recall
    b = det_precision + det_recall
    det_f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    det_threshold = save_scores(c, det_recall, det_precision, det_f1, det_thresholds, txt_file, 'det', save_dir) 
    print('Optimal DET Threshold: {:.2f}'.format(det_threshold))
    export_hist(c, gt_label, score_label, 'det', save_dir)

    file_list = test_dataset.x

    if c.th_manual >0:
        seg_threshold = c.th_manual
        print('Manual PIX Threshold: {:.2f}'.format(seg_threshold))
    else:
        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), super_mask.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        seg_threshold = save_scores(c, recall, precision, f1, thresholds, txt_file, 'seg', save_dir) 
        print('Optimal PIX Threshold: {:.2f}'.format(seg_threshold))

    export_test_images(c, test_img, gt_mask, super_mask, pix_image, feature_maps, seg_threshold, file_list, save_dir, txt_file, w_fe, result_dict)
    export_hist(c, gt_mask, super_mask, 'pix', save_dir)
    txt_file.close()


# calculate AUPRO metric
# This function is from the CDO project (https://github.com/caoyunkang/CDO/tree/master)  
def cal_pro_metric(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=200):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool)
    for step in range(max_steps):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    return pro_auc_score

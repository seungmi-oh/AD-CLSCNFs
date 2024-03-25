'''This code is based on the DRAEM project (source: https://github.com/VitjanZ/DRAEM).
We modified and added the necessary modules or functions for our purposes.'''
import os, copy
import numpy as np
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
from torchvision import transforms as T
from .perlin import rand_perlin_2d_np

reverse_mask_cls = ['cable', 'hazelnut', 'metal_nut', 'pill', 'toothbrush', '01', '03']
no_bg_use_cls = ['transistor', 'tile', 'wood', 'carpet', 'leather', 'total', '02']
no_fill_hole_cls = ['grid', 'metal_nut', '01']

class GenerateSyntheticTrainDataset(Dataset):

    def __init__(self, root_dir, class_name, dataset, norm_mean, norm_std, aug_ratio, use_in_domain_data, resize_shape=None, for_check =False):
        """
        Args:
            root_dir (string): Directory with all the images (DTD, MVTecAD, and BTAD dataset).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.class_name = class_name

        # in-domain dataset
        if dataset =='mvtec':
            img_dir = os.path.join(root_dir, dataset, class_name, 'train', 'good')
        if dataset =='btad':
            img_dir = os.path.join(root_dir, dataset, class_name, 'train', 'ok')

        self.image_paths = sorted([os.path.join(img_dir, f)                   
                                       for f in os.listdir(img_dir)
                                       if f.endswith('.png') or f.endswith('.bmp')])
        
        # DTD dataset to genrate apparent anomalies 
        anomaly_source_dir = os.path.join(os.path.join(root_dir, 'dtd'), 'images') 
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_dir+"/*/*.jpg"))


        # set transformations applied to images
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      ]

        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.normalize = T.Compose([T.Normalize(norm_mean, norm_std)])

        self.aug_ratio =aug_ratio
        self.use_in_domain_data = use_in_domain_data

        # set different binarization process depending on classes to remove defects floating on the background 
        if class_name in reverse_mask_cls:
            self.reverse = True
        else:
            self.reverse = False
        if class_name in no_fill_hole_cls:
            self.fill_holes = False
        else:
            self.fill_holes = True
        
        # if for_check is True --> check the synthetic defect generation process
        # else --> generate the synthetic defects to train a network 
        self.for_check = for_check

    def __len__(self):
        return len(self.image_paths)


    # randomly transform the anomaly source image.
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    # get the object mask to remove defects floating on the background
    def detect_object_area(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (3,3), 0)
        _,mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if self.reverse==False:
            mask = 255 - mask
        else:
            pass

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = cv2.GaussianBlur(mask, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

        mask = (2*(mask.astype(np.float32))-255.0).clip(0,255).astype(np.uint8)

        if self.fill_holes == True:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for i in range(len(contours)):
                obj_mask = cv2.fillPoly(mask, [contours[i]], color=(255,255,255))
        else:
            obj_mask = mask

        bg_img = img.copy()
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)
        bg_img[:, :, 3] = np.uint8(255-obj_mask)

        bg_removed_img = img.copy()
        bg_removed_img = cv2.cvtColor(bg_removed_img, cv2.COLOR_BGR2BGRA)
        bg_removed_img[:, :, 3] = np.uint8(obj_mask)
        return bg_img, bg_removed_img, obj_mask

    # generate synthetic defect data 
    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        anomaly_source_img = cv2.cvtColor(anomaly_source_img, cv2.COLOR_BGR2RGB)
        anomaly_img_augmented = aug(image=anomaly_source_img)

        # get object mask
        bg_img, bg_removed_img, obj_mask = self.detect_object_area(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        org_image = copy.deepcopy(image)
        obj_mask_augmented = copy.deepcopy(obj_mask/255)

        # get an irregular pattern mask 
        perlin_scale = 6
        min_perlin_scale = 0
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        # get anomaly samples
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        # blending two images (input in-domain data and anomaly samples)
        beta = torch.rand(1).numpy()[0] * 0.8
        blended_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
        blended_image = blended_image.astype(np.float32)

        if (self.class_name in no_bg_use_cls)==True:
            msk = perlin_thr
        else: # remove defects floating on the background 
            msk = perlin_thr *np.expand_dims(obj_mask_augmented, axis=2)
        msk[msk>0]=1
        msk = msk.astype(np.float32)

        # get the number of synthetic anomaly segments 
        defect_seg_nums, defect_seg_labels, _, _ = cv2.connectedComponentsWithStats(np.uint8(msk))

        no_anomaly = torch.rand(1).numpy()[0]
        if self.for_check ==False:
            if no_anomaly > self.aug_ratio or defect_seg_nums==1: # randomly do not generate synthetic defect to train good samples (1-aug_ratio)*100 %
                image = image.astype(np.float32)
                return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0],dtype=np.float32)
            else: # randomly generate synthetic defect data
                # smoothing the ground truth probability of synthetic defects according to color differences 
                diff = torch.FloatTensor(msk*np.expand_dims(np.mean(blended_image-org_image, -1),-1))
                diff[diff==0] = -100
                smooth_msk = torch.sigmoid(diff*5)
                smooth_msk[diff==-100] =0
                smooth_msk = smooth_msk.detach().numpy() 
                msk[msk>0] = np.mean(smooth_msk[smooth_msk>0])

                # There is no anomalies in the synthetic defect image since the defects are generated in the background only. 
                if np.sum(msk) == 0:
                    has_anomaly=0.0
                    msk = np.zeros(msk.shape)
                    augmented_image = org_image
                # get the final synthetic defect image. 
                else:
                    one_hot_msk = np.uint8(msk>0)
                    augmented_image = one_hot_msk.astype(np.float32) * blended_image + (1-one_hot_msk.astype(np.float32))*image
                    has_anomaly = 1.0
                return augmented_image, msk, np.array([has_anomaly],dtype=np.float32)
        else:
            if no_anomaly > self.aug_ratio or defect_seg_nums==1:
                msk = np.zeros_like(perlin_thr, dtype=np.float32)
                defect_samples = msk 
                augmented_image = image
            else:
                # smoothing the ground truth probability of synthetic defects according to color differences 
                diff = torch.FloatTensor(msk*np.expand_dims(np.mean(blended_image-org_image, -1),-1))
                diff[diff==0] = -100
                smooth_msk = torch.sigmoid(diff*5)
                smooth_msk[diff==-100] =0
                smooth_msk = smooth_msk.detach().numpy() 
                msk[msk>0] = np.mean(smooth_msk[smooth_msk>0])

            # There is no anomalies in the synthetic defect image since the defects are generated in the background only. 
            if np.sum(msk) == 0: 
                has_anomaly=0.0
                msk = np.zeros(msk.shape)
                defect_samples = msk 
                augmented_image = org_image
            # get the final synthetic defect image. 
            else:
                has_anomaly = 1.0
                one_hot_msk = np.uint8(msk>0)
                defect_samples = one_hot_msk.astype(np.float32)* blended_image 
                augmented_image = one_hot_msk.astype(np.float32) * blended_image + (1-one_hot_msk.astype(np.float32))*image
            return org_image, perlin_thr, anomaly_source_img, anomaly_img_augmented, obj_mask, img_thr, defect_samples, msk, augmented_image 

    # generate synthetic anomalies
    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))

        if self.for_check ==False:
            augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
            augmented_image = np.transpose(augmented_image, (2, 0, 1))
            image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
            return image, augmented_image, anomaly_mask, has_anomaly
        else:
            org_image, perlin_thr, anomaly_source_img, anomaly_img_augmented, obj_mask, img_thr, defect_samples, msk, augmented_image  = self.augment_image(image, anomaly_source_path)
            return org_image*255, perlin_thr*255, anomaly_source_img, anomaly_img_augmented, obj_mask, img_thr*255, defect_samples*255, msk*255, augmented_image*255 

    def __getitem__(self, idx):
        if self.for_check ==False: 
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            if self.use_in_domain_data==False:
                # randomly select anomaly source image from DTD data
                image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx])
            else:
                # randomly select anomaly source image from in-domain data and DTD data
                in_domain = torch.rand(1).numpy()[0]
                if in_domain > 0.5:
                    anomaly_idx = torch.randint(0, len(self.image_paths), (1,)).item()
                    image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx], self.image_paths[anomaly_idx])
                else:
                    image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx])

            augmented_image = torch.FloatTensor(augmented_image)
            augmented_image = self.normalize(augmented_image)
            anomaly_mask = torch.FloatTensor(anomaly_mask)
            augmented_image = augmented_image.type(torch.float32)
            anpmaly_mask = anomaly_mask.type(torch.float32)
            return  augmented_image, has_anomaly, anomaly_mask
        else:
            anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
            if self.use_in_domain_data==False:
                # randomly select anomaly source image from DTD data
                return self.transform_image(self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx])
            else:
                # randomly select anomaly source image from in-domain data and DTD data
                in_domain = torch.rand(1).numpy()[0]
                if in_domain > 0.5:
                    anomaly_idx = torch.randint(0, len(self.image_paths), (1,)).item()
                    return self.transform_image(self.image_paths[idx], self.image_paths[anomaly_idx])
                else:
                    return self.transform_image(self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx])

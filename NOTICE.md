NOTICE for [Your Project Name]

[Your Project Name]

Copyright (c) 2024 Seungmi Oh


This product includes software developed by the followings:

Original Codes

1. CFlow-AD
    - Project: CFlow-AD (Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows)
    - URL: https://github.com/gudovskiy/cflow-ad/tree/master
    
    Copyright (c) 2021, Panasonic AI Lab of Panasonic Corporation of North America.
  
    License: BSD 3-Clause (https://opensource.org/license/bsd-3-clause/)
  
    See the original LICENSE file for details.
  
    We implemented our method based on this software.
  
    Portions of this software have been modified for our purposes by Seungmi Oh.

2. DRAEM
    - Project: DRAEM (A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection)
    - URL: https://github.com/VitjanZ/DRAEM
  
    Copyright (c) 2021 VitjanZ
  
    License: MIT (https://opensource.org/licenses/MIT)
  
    See the original LICENSE file for details.
  
    "perlin.py" of this software is used for generating the synthetic defect data. 
  
    "data_loader.py" of this software have been modified for our purposes by Seungmi Oh.
  

3. CDO
    - Project: CDO (Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization)
    - URL: https://github.com/caoyunkang/CDO
  
    Copyright (c) 2023 Yunkang Cao
  
    License: MIT (https://opensource.org/licenses/MIT)
  
    See the original LICENSE file for details.
  
    "cal_pro_metric" function of this software is used for calcuating AUPRO (Area Under Per-Region Overlap) metric.


Modifications

- Description: We implemented our method using the codes of CFlow-AD project, but we have been modified the most parts of the codes for our purpses. 

  We also add some codes or modules to train the pixel-wise classification network and aggregating the score maps of the pixel-wise classification network and CNF networks.

- Copyright (c) 2024 Seungmi Oh

# Anomaly Detection Using Normalizing Flow-Based Density Estimation and Self-Supervising Pixelwise Classification with a Shared Feature Extractor

## Abstract
Anomaly detection (AD) aims to detect instances containing patterns that are significantly distinct from normal data encountered during training. Recent studies on distance-based methods for anomaly detection have utilized a pretrained network on a large-scale dataset (e.g., ImageNet) to extract normal features for industrial inspection. However, because of the large domain gap, the network cannot extract intrinsic features from the inspection data. Moreover, simulation-based methods for anomaly detection generate and use synthetic abnormal data to improve discriminative ability. Nevertheless, this can cause a problem of overfitting to synthetic abnormal appearances. Considering these problems, we propose a method that combines a pixelwise classification network and conditional normalizing flow (CNF) networks by sharing feature extractors to enhance performance. We also propose a hybrid training algorithm to induce collaborative effects of these two networks. The pixelwise classification network finetunes the pretrained feature extractor of CNF networks using synthetic abnormal data to learn the discriminative and intrinsic features of the in-domain data. Subsequently, we train CNF networks with the finetuned feature extractor using only normal data for density estimation, thereby mitigating the overfitting problem of the pixelwise classification network. The final prediction is obtained by the weighted averaging of the predictions of the two networks to maintain their advantages and alleviate their disadvantages.

## BibTex Citation

## Quick Start
<details>
<summary>
  ### Install
</summary>
  
    git clone https://github.com/meitu
    cd CLS_NF_shareFE
    python3 -m pip install -U -r requirements.txt
   
</details>

<details>
<summary>
  ### Prepare Dataset
</summary>
  
    bash run_scripts/construct_dataset.sh
    
</details>


<details>
<summary>
  ### Train Networks 
</summary>
  - Train and evaluate our network and CFlow-AD for all categories  
  
    bash run_scripts/train_eval_total.sh

  - Train and evaluate our network and CFlow-AD by selecting class_name (ex. bottle)
  
    bash run_scripts/mvtec/bottle/train_eval_total.sh

</details>

# Anomaly Detection Using Normalizing Flow-Based Density Estimation and Self-Supervising Pixelwise Classification with a Shared Feature Extractor
This is the pytorch implementation to the 2023 IEEE Access paper "Anomaly Detection Using Normalizing Flow-Based Density Estimation and Self-Supervising Pixelwise Classification with a Shared Feature Extractor" by Seungmi Oh and Jeongtae Kim.

## Abstract
Anomaly detection (AD) aims to detect instances containing patterns that are significantly distinct from normal data encountered during training. Recent studies on distance-based methods for anomaly detection have utilized a pretrained network on a large-scale dataset (e.g., ImageNet) to extract normal features for industrial inspection. However, because of the large domain gap, the network cannot extract intrinsic features from the inspection data. Moreover, simulation-based methods for anomaly detection generate and use synthetic abnormal data to improve discriminative ability. Nevertheless, this can cause a problem of overfitting to synthetic abnormal appearances. Considering these problems, we propose a method that combines a pixelwise classification network and conditional normalizing flow (CNF) networks by sharing feature extractors to enhance performance. We also propose a hybrid training algorithm to induce collaborative effects of these two networks. The pixelwise classification network finetunes the pretrained feature extractor of CNF networks using synthetic abnormal data to learn the discriminative and intrinsic features of the in-domain data. Subsequently, we train CNF networks with the finetuned feature extractor using only normal data for density estimation, thereby mitigating the overfitting problem of the pixelwise classification network. The final prediction is obtained by the weighted averaging of the predictions of the two networks to maintain their advantages and alleviate their disadvantages.

## BibTex Citation

## Quick Start
<details>
<summary>
Install
</summary>
  
    git clone https://github.com/meitu
    cd CLS_NF_shareFE
    python3 -m pip install -U -r requirements.txt
   
</details>

<details>
<summary>
Prepare Dataset
</summary>
- We used [MVTec AD]<https://www.mvtec.com/company/research/datasets/mvtec-ad/> dataset to train and inference networks for anomaly detection and localization for quality inspection in Industry. 
  We also generated synthetic defect data using the [DTD]<https://www.robots.ox.ac.uk/~vgg/data/dtd/> dataset to finetune a feature extractor of CNF networks by training the pixel-wise classification network.
    
- Using the command below, you can automatically download MVTecAD dataset and DTD dataset at the parent directory of the project directory. 
  Also, the command generates and saves a synthetic defect validation dataset at the parent directory of the project directory. 
  
    bash run_scripts/construct_dataset.sh
    
</details>


<details>
<summary>
Train and Evaluate Networks 
</summary>
  
- Train and evaluate our network and CFlow-AD for all categories  
    
      bash run_scripts/train_eval_total.sh
      
- Train and evaluate our network and CFlow-AD by selecting class_name (ex. bottle)
    
      bash run_scripts/mvtec/bottle/train_eval_total.sh

</details>

<details>
<summary>
Inference Our Reference MVTec Results
</summary>
  
- Download checkpoints

- We trained every models three times with random initialization to avoid over-estimation by each model. 

- Among three experimental results, we seleced the best results for each category and uploaded the models at the google drive.

- You can download checkpoints of the models using the command below. 
    
      bash run_scripts/download_best_models.sh
      
- Evaluate the best models of the proposed method and CFlow-AD for all categories and parse results
    
      bash run_scripts/eval_best_models.sh
      
- Evaluate the best models of the proposed method and CFlow-AD by selecting class_name (ex. bottle)
    
      bash run_scripts/mvtec/bottle/eval_best_models.sh

</details>


## Architecture


## The Quantiative Results
<details>
<summary>
Reference Results for MVTec (Averaged on three runs with different random initialization)
</summary>


</details>

<details>
<summary>
Best Results for MVTec (Results of the uploaded checkpoint)
</summary>


</details>


## Credits


## License

This project is licensed under the MIT License.



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
- We also generated synthetic defect data using the [DTD]<https://www.robots.ox.ac.uk/~vgg/data/dtd/> dataset to finetune a feature extractor of CNF networks by training the pixel-wise classification network.  
- Using the command below, you can automatically download MVTecAD dataset and DTD dataset at the parent directory of the project directory.
- Also, the command generates and saves a synthetic defect validation dataset at the parent directory of the project directory. 
  
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
  
| Method            | PaDiM                                   || Cflow-AD                                || DRAEM                                   || CDO                                     || Proposed Method                                |
|-------------------|:---------:|:---------:|:--------:|:-----:|:---------:|:---------:|:--------:|:-----:|:---------:|:---------:|:--------:|:-----:|:---------:|:---------:|:--------:|:-----:|:---------------:|:---------:|:--------:|:-----:|
| Metric \ Category | Img AUROC | Pix AUROC | Pix AUPR | AUPRO | Img AUROC | Pix AUROC | Pix AUPR | AUPRO | Img AUROC | Pix AUROC | Pix AUPR | AUPRO | Img AUROC | Pix AUROC | Pix AUPR | AUPRO | Img AUROC       | Pix AUROC | Pix AUPR | AUPRO |
| Bottle            | 100.00    | 98.71     | 72.79    | 94.48 | 100.00    | 98.74     | 73.54    | 94.48 | 99.73     | 99.27     | 89.90    | 96.27 | 100.00    | 99.18     | 86.28    | 96.51 | 99.89           | 99.12     | 88.06    | 96.60 |
| Cable             | 95.20     | 97.28     | 59.45    | 93.22 | 93.82     | 97.28     | 59.07    | 93.26 | 90.70     | 95.13     | 62.90    | 76.27 | 92.57     | 96.96     | 61.40    | 94.36 | 96.09           | 97.33     | 59.66    | 93.38 |
| Capsule           | 97.81     | 99.06     | 48.88    | 94.47 | 97.22     | 99.06     | 49.42    | 94.50 | 94.00     | 92.50     | 45.70    | 86.83 | 83.10     | 98.51     | 41.78    | 93.42 | 98.56           | 99.21     | 58.48    | 95.49 |
| Carpet            | 98.48     | 99.24     | 66.38    | 96.76 | 98.26     | 99.24     | 66.23    | 96.77 | 85.53     | 95.70     | 60.00    | 90.07 | 97.50     | 98.98     | 56.70    | 95.58 | 99.44           | 99.42     | 78.65    | 97.95 |
| Grid              | 98.50     | 98.89     | 37.98    | 95.79 | 98.80     | 98.89     | 37.95    | 95.78 | 99.87     | 99.53     | 57.47    | 97.47 | 96.30     | 98.82     | 42.69    | 96.24 | 99.78           | 99.11     | 53.97    | 96.59 |
| Hazelnut          | 100.00    | 98.81     | 62.19    | 96.79 | 99.99     | 98.81     | 62.12    | 96.71 | 99.27     | 99.60     | 89.00    | 98.50 | 98.74     | 99.15     | 66.80    | 97.18 | 96.55           | 99.21     | 79.25    | 97.92 |
| Leather           | 100.00    | 99.60     | 57.99    | 98.88 | 100.00    | 99.59     | 57.99    | 98.90 | 99.70     | 98.97     | 71.13    | 97.37 | 100.00    | 99.62     | 61.64    | 98.69 | 100.00          | 99.79     | 78.09    | 99.49 |
| Metal Nut         | 99.32     | 97.92     | 78.53    | 94.39 | 99.17     | 97.93     | 78.62    | 94.27 | 99.00     | 98.73     | 91.07    | 93.57 | 98.14     | 98.34     | 83.15    | 94.95 | 98.22           | 97.89     | 84.12    | 95.84 |
| Pill              | 95.85     | 98.43     | 71.46    | 96.03 | 95.50     | 98.43     | 71.71    | 95.98 | 96.67     | 97.30     | 45.63    | 85.43 | 96.84     | 98.45     | 79.75    | 97.06 | 97.70           | 98.93     | 85.35    | 96.90 |
| Screw             | 92.38     | 98.71     | 36.02    | 94.34 | 92.04     | 98.68     | 35.05    | 94.26 | 98.80     | 99.57     | 70.60    | 95.40 | 83.05     | 98.86     | 31.48    | 93.97 | 94.52           | 99.04     | 52.43    | 95.38 |
| Tile              | 99.60     | 97.67     | 77.01    | 91.23 | 99.64     | 97.67     | 76.89    | 91.31 | 100.00    | 99.43     | 96.47    | 98.17 | 99.78     | 97.64     | 67.02    | 92.02 | 99.31           | 99.21     | 93.07    | 96.71 |
| Toothbrush        | 84.17     | 98.37     | 33.33    | 90.12 | 84.35     | 98.37     | 33.05    | 90.16 | 97.87     | 98.50     | 54.43    | 90.73 | 86.94     | 98.80     | 43.60    | 91.25 | 94.07           | 98.45     | 45.87    | 89.67 |
| Transistor        | 96.58     | 90.94     | 50.37    | 82.71 | 97.03     | 91.09     | 50.40    | 82.84 | 90.30     | 86.37     | 47.70    | 75.97 | 93.47     | 85.44     | 49.01    | 75.09 | 98.51           | 95.17     | 62.22    | 89.53 |
| Wood              | 96.32     | 95.79     | 57.34    | 93.03 | 95.79     | 95.80     | 57.23    | 93.16 | 99.33     | 97.10     | 79.13    | 91.40 | 99.18     | 97.29     | 65.77    | 94.73 | 99.68           | 97.54     | 81.06    | 96.33 |
| Zipper            | 98.82     | 99.00     | 56.28    | 96.55 | 98.99     | 99.01     | 56.40    | 96.55 | 97.87     | 98.70     | 74.70    | 94.97 | 97.00     | 97.99     | 54.32    | 94.17 | 99.10           | 99.39     | 81.08    | 97.55 |
| Average           | 96.87     | 97.89     | 57.73    | 93.92 | 96.71     | 97.91     | 57.71    | 93.93 | 96.58     | 97.09     | 69.06    | 91.23 | 94.84     | 97.60     | 59.42    | 93.68 | 98.09           | 98.59     | 72.09    | 95.69 |

</details>

<details>
<summary>
Best Results for MVTec (Results of the uploaded checkpoint)
</summary>


</details>


## Credits


## License

This project is licensed under the MIT License.



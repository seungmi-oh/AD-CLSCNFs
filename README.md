# Anomaly Detection Using Normalizing Flow-Based Density Estimation and Synthetic Defect Classification
This is the pytorch implementation to the 2024 IEEE Access paper "Anomaly Detection Using Normalizing Flow-Based Density Estimation and Synthetic Defect Classification" by Seungmi Oh and Jeongtae Kim.

## Abstract
We propose a novel deep learning-based anomaly detection (AD) system that combines a pixelwise classification network with conditional normalizing flow (CNF) networks by sharing feature extractors. We trained the pixelwise classification network using synthetic abnormal data to fine-tune a pretrained feature extractor of the CNF networks, thereby learning the discriminative features of the in-domain data. After that, we trained the CNF networks using normal data with the fine-tuned feature extractor to estimate the density of normal data. During inference, we detect anomalies by calculating the weighted average of the anomaly scores from the pixelwise classification and CNF networks. Because the proposed system not only has learned the properties of in-domain data but also aggregated the anomaly scores of the classification and CNF networks, it showed significantly improved performance compared to existing methods in experiments using the MvTecAD and BTAD datasets. Moreover, the proposed system does not increase computations intensively since the classification and the density estimation systems share feature extractors.

## BibTex Citation

## Quick Start
<details>
<summary>
Settings
</summary>
  
- OS: Ubuntu 20.04.1 LTS
- Language: python 3.8.10
- Other dependencies in requirements.txt or Pipfile.lock

      git clone https://github.com/seungmi-oh/AD-CLSCNFs.git
      mv AD-CLSCNFs codes
      cd codes
  
      # create and activate a virtual environment using virtualenv or pipenv
      python3 -m pip install -U -r requirements.txt # virtualenv
      pipenv install Pipfile # pipenv
   
</details>

<details>
<summary>
Prepare Dataset
</summary>
  
- We used [MVTec AD (MVTec Anomaly Detection)](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and [BTAD (Bean-Tech Anomaly Detection)](http://avires.dimi.uniud.it/papers/btad/btad.zip) datasets to train and inference networks for anomaly detection and localization for quality inspection in Industry.
- We also generated synthetic defect data using the [DTD (Describable Textures Dataset)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) to finetune a feature extractor of CNF networks by training the pixel-wise classification network.  
- Using the command below, you can automatically download MVTecAD dataset and DTD dataset at the parent directory of the project directory.
- Also, the command generates and saves a synthetic defect validation dataset at the parent directory of the project directory. 
  
      bash run_scripts/construct_dataset.sh
    
</details>


<details>
<summary>
Train and Evaluate Networks 
</summary>
  
- Train and evaluate our network and CFlow-AD for the MVTecAD and BTAD datasets

      # all categories of the MVTecAD dataset
      bash run_scripts/mvtec/train_eval_total.sh
  
      # all products of the BTAD dataset
      bash run_scripts/btad/train_eval_total.sh
  
      
- Train and evaluate our network and CFlow-AD by selecting class_name (ex. bottle/01)
    
      # the category 'bottle' of the MVTecAD dataset
      bash run_scripts/mvtec/bottle/train_eval_total.sh
  
      # the product '01' of the BTAD dataset
      bash run_scripts/btad/01/train_eval_total.sh

</details>

<details>
<summary>
Inference Our Models Shown The Best Performance for The MVTecAD and BTAD Datasets.
</summary>
  
- Download checkpoints
- We trained every models three times with random initialization to avoid over-estimation by each model. 
- Among three experimental results, we seleced the best results for each category and uploaded the models at [the link](https://www.dropbox.com/scl/fi/wryllmczt0y1syf0a7o9d/best_models.zip?rlkey=hvlrovalojf15goo5vp142mue).
- You can download checkpoints of the models using the command below. 
    
      bash run_scripts/download_best_models.sh
      
- Evaluate the best models of the proposed method and CFlow-AD 

      # all categories of the MVTecAD dataset
      bash run_scripts/mvtec/eval_best_models.sh
  
      # all products of the BTAD dataset
      bash run_scripts/btad/eval_best_models.sh
      
- Evaluate the best models of the proposed method and CFlow-AD by selecting class_name (ex. bottle/01)

      # the category 'bottle' of the MVTecAD dataset
      bash run_scripts/mvtec/bottle/eval_best_model.sh
  
      # the product '01' of the BTAD dataset
      bash run_scripts/btad/01/eval_best_model.sh

</details>


## Network Architecture
We propose a novel deep learning-based AD system that combines a pixelwise classification network with conditional normalizing flow networks by sharing feature extractors. The proposed system showed the satisfactory performance thanks to the discriminative features of in-domain data and the positive impact of network ensembles.

![graphical_abstract](https://github.com/seungmi-oh/AD-CLSCNFs/assets/141846117/49a63ad4-8603-4e30-8946-648ed9d8eb74)

## The Quantiative Results
<details>
<summary>
Reference Results for The MVTecAD and BTAD Datasets (Averaged on three runs with different random initialization)
</summary>
  
- MVTecAD dataset
  
| Category   \  Metric | Img AUROC | Pix AUROC |  Pix AUPR |   AUPRO   |
|----------------------|:---------:|:---------:|:---------:|:---------:|
|        Bottle        |   99.89   |   99.12   |   88.06   |   96.60   |
|         Cable        |   96.09   |   97.33   |   59.66   |   93.38   |
|        Capsule       |   98.56   |   99.21   |   58.48   |   95.49   |
|        Carpet        |   99.44   |   99.42   |   78.65   |   97.95   |
|         Grid         |   99.78   |   99.11   |   53.97   |   96.59   |
|       Hazelnut       |   96.55   |   99.21   |   79.25   |   97.92   |
|        Leather       |   100.00  |   99.79   |   78.09   |   99.49   |
|       Metal Nut      |   98.22   |   97.89   |   84.12   |   95.84   |
|         Pill         |   97.70   |   98.93   |   85.35   |   96.90   |
|         Screw        |   94.52   |   99.04   |   52.43   |   95.38   |
|         Tile         |   99.31   |   99.21   |   93.07   |   96.71   |
|      Toothbrush      |   94.07   |   98.45   |   45.87   |   89.67   |
|      Transistor      |   98.51   |   95.17   |   62.22   |   89.53   |
|         Wood         |   99.68   |   97.54   |   81.06   |   96.33   |
|        Zipper        |   99.10   |   99.39   |   81.08   |   97.55   |
|      **Average**     | **98.09** | **98.59** | **72.09** | **95.69** |

- BTAD dataset

|  Product   \  Metric | Img AUROC | Pix AUROC |  Pix AUPR |   AUPRO   |
|----------------------|:---------:|:---------:|:---------:|:---------:|
| 01                   |   99.48   |   45.99   |   95.27   |   63.60   |
| 02                   |   88.66   |   66.33   |   96.61   |   57.37   |
| 03                   |   99.78   |   48.97   |   99.49   |   96.99   |
| **Average**          | **95.97** | **53.76** | **97.12** | **72.66** |
  
  
</details>

<details>
<summary>
Best Results for The MVTecAD and BTAD Datasets (Results of the uploaded checkpoint)
</summary>
  
- MVTecAD dataset
  
| Category   \  Metric | Img AUROC | Pix AUROC |  Pix AUPR |   AUPRO   |
|----------------------|:---------:|:---------:|:---------:|:---------:|
| Bottle               |   100.00  |   89.96   |   99.25   |   97.01   |
| Cable                |   94.17   |   60.59   |   97.38   |   93.44   |
| Capsule              |   98.96   |   61.03   |   99.28   |   95.63   |
| Carpet               |   99.72   |   79.56   |   99.44   |   98.08   |
| Grid                 |   100.00  |   58.43   |   99.14   |   96.91   |
| Hazelnut             |   99.79   |   79.63   |   99.22   |   97.94   |
| Leather              |   100.00  |   81.22   |   99.81   |   99.56   |
| Metal Nut            |   98.58   |   87.91   |   98.37   |   96.32   |
| Pill                 |   97.41   |   88.11   |   99.07   |   96.67   |
| Screw                |   96.86   |   56.17   |   99.33   |   96.62   |
| Tile                 |   100.00  |   94.42   |   99.33   |   97.23   |
| Toothbrush           |   98.89   |   49.78   |   98.50   |   90.38   |
| Transistor           |   98.75   |   64.52   |   95.92   |   90.30   |
| Wood                 |   99.82   |   81.79   |   97.68   |   96.40   |
| Zipper               |   99.74   |   81.63   |   99.41   |   97.66   |
| **Average**          | **98.85** | **74.32** | **98.74** | **96.01** |

- BTAD dataset

|  Product   \  Metric | Img AUROC | Pix AUROC |  Pix AUPR |   AUPRO   |
|----------------------|:---------:|:---------:|:---------:|:---------:|
| 01                   |   99.42   |   46.56   |   95.28   |   64.08   |
| 02                   |   88.53   |   67.19   |   96.63   |   57.38   |
| 03                   |   99.83   |   53.84   |   99.49   |   96.84   |
| **Average**          | **95.93** | **55.86** | **97.13** | **72.77** |

</details>


## Credits

We implemented our method using some portions of the codes of [CFlow-AD](https://github.com/gudovskiy/cflow-ad), [DRAEM](https://github.com/VitjanZ/DRAEM), and [CDO](https://github.com/caoyunkang/CDO) projects.  
We added a NOTICE file (NOTICE.md) to give credits the authors of the projects. 

## License

This project is licensed under [the MIT License](https://opensource.org/license/mit/).



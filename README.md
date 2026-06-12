## Attribute-Guidance and Size-Aware Learning for Fine-Grained Blood Cell Classification

### Environment
Our code is based on [MMPreTrain](https://github.com/open-mmlab/mmpretrain).

In this work, we used:
- python=3.10.16
- pytorch=1.13.1
- mmpretrain=1.2.0
- mmcv=2.2.0

The environment can be installed with:
```
conda create -n AGSA python=3.10
conda activate AGSA
pip install -r requirements.txt
pip install -e .
chmod u+x tools/*
```

### Datasets
We carried out experiments on two datasets, i.e., [Chula](https://github.com/Chula-PIC-Lab/Chula-RBC-12-Dataset) and BCFC-329k dataset. Please organize the structure of datasets like this:
```
data
|——Chula
|  |——train
|  |  |——class_0
|  |  |	 |——Chula_01.jpg
|  |  |	 |——...
|  |  |  |——Chula_100.jpg
|  |  |——class_1
|  |  |——...
|  |  |——class_12
|  |——val
|  |  |——class_0
|  |  |——class_1
|  |  |——...
|  |  |——class_12
|  |——test
|  |  |——class_0
|  |  |——class_1
|  |  |——...
|  |  |——class_12
|——BCFC-329k
|  |——train
|  |  |——class_0
|  |  |	 |——BCFC_01.jpg
|  |  |	 |——...
|  |  |  |——BCFC_100.jpg
|  |  |——class_1
|  |  |——...
|  |  |——class_46
|  |——val
|  |  |——class_0
|  |  |——class_1
|  |  |——...
|  |  |——class_46
|  |——test
|  |  |——class_0
|  |  |——class_1
|  |  |——...
|  |  |——class_46
```

### Training

To train the AGSA or AGSA-distilled model on the BCFC-329k dataset, you can use commands like this:
```
CUDA_VISIBLE_DEVICES=0 python TFR_MFA/BCFC-329k/0-extract_attribute_text_feature.py

CUDA_VISIBLE_DEVICES=0 python TFR_MFA/BCFC-329k/1-train_TFR.py

CUDA_VISIBLE_DEVICES=0 python TFR_MFA/BCFC-329k/2-extract_all_raw_image_embeddings.py

CUDA_VISIBLE_DEVICES=0 python TFR_MFA/BCFC-329k/3-train_MFA.py

CUDA_VISIBLE_DEVICES=0,1 PORT=29500 tools/dist_train.sh configs/BCFC-329k/AGSA.py 2
or
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 tools/dist_train.sh configs/BCFC-329k/AGSA_distilled.py 2
```

The trained model will be stored to `work_dirs/`.

In this project, we provide two implementations: AGSA and AGSA-distilled.
AGSA is the full model used in our paper, which combines attribute guidance and size-aware learning for fine-grained blood cell classification.
AGSA-distilled is a distilled version of AGSA. It replaces the VLM image branch with a lightweight MoE module, reducing parameters and computational overhead while maintaining competitive performance.

### Models and Evaluation

We evaluate our method on Chula and BCFC-329k test set.

| Dataset  | F1 | Precision | Recall |download                                                                                                                                                       |
| ------- | :----: | :----: | :----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chula |  85.64 |  88.55 | 83.39 | [config](https://github.com/CVIU-CSU/AGSA/blob/main/configs/Chula/AGSA.py); [model](https://drive.google.com/file/d/1e3CqHqKJG0-D32MmuFo5NUwZv33MU1vE/view?usp=sharing) |
| BCFC-329k |  79.18 |  80.25 | 79.24 | [config](https://github.com/CVIU-CSU/AGSA/blob/main/configs/BCFC-329k/AGSA.py); [model](https://drive.google.com/file/d/1U-MwRr0fnXgYRzpdwLtT2it59mBWT6Xz/view?usp=sharing) |

For calculating FLOPs and Params, you can use the following commands:
```
CUDA_VISIBLE_DEVICES=0 python tools/get_flops_v2.py configs/BCFC-329k/AGSA_distilled.py
```

### Acknowledgements
Last, we thank these authors for sharing their ideas and implementations:
- [MMPreTrain](https://github.com/open-mmlab/mmpretrain)
- [WBCAtt](https://github.com/apple2373/wbcatt)
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)

This manuscript was supported in part by the Natural Science Foundation of China under Grant 62473385, the Natural Science Foundation of Hunan Province, China under Grant 2024JJ5444, and the Changsha Municipal Natural Science Foundation under Grant kq2402227. 
The authors acknowledge the High Performance Computing Center of Central South University for computational resources.

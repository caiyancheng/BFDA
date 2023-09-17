[![IEEE](https://img.shields.io/badge/IEEE-10231122-b31b1b.svg)](https://ieeexplore.ieee.org/document/10231122)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/caiyancheng/BFDA/pulls)

# T-IP-2023: Rethinking Cross-Domain Pedestrian Detection: A Background-Focused Distribution Alignment Framework for Instance-Free One-Stage Detectors

### Abstract
Cross-domain pedestrian detection aims to generalize pedestrian detectors from one label-rich domain to another label-scarce domain, which is crucial for various real-world applications. Most recent works focus on domain alignment to train domain-adaptive detectors either at the instance level or image level. From a practical point of view, one-stage detectors are faster. Therefore, we concentrate on designing a cross-domain algorithm for rapid one-stage detectors that lacks instance-level proposals and can only perform image-level feature alignment. However, pure image-level feature alignment causes the foreground-background misalignment issue to arise, i.e., the foreground features in the source domain image are falsely aligned with background features in the target domain image. To address this issue, we systematically analyze the importance of foreground and background in image-level cross-domain alignment, and learn that background plays a more critical role in image-level cross-domain alignment. Therefore, we focus on cross-domain background feature alignment while minimizing the influence of foreground features on the cross-domain alignment stage. This paper proposes a novel framework, namely, background-focused distribution alignment (BFDA), to train domain adaptive onestage pedestrian detectors. Specifically, BFDA first decouples the background features from the whole image feature maps and then aligns them via a novel long-short-range discriminator. Extensive experiments demonstrate that compared to mainstream domain adaptation technologies, BFDA significantly enhances cross-domain pedestrian detection performance for either one-stage or two-stage detectors. Moreover, by employing the efficient one-stage detector (YOLOv5), BFDA can reach 217.4 FPS (640×480 pixels) on NVIDIA Tesla V100 (7∼12 times the FPS of the existing frameworks), which is highly significant for practical applications.

&ensp;
<p align="center">
  <img src="./images/Figure_1.png" width="100%" height="420">
</p>

### Datasets
The primary datasets employed in this paper consist of Cityscapes, Caltech, and Foggycityscapes. Below, we present the Cityscapes and Caltech datasets used in this study:
* [Cityscapes](https://drive.google.com/drive/folders/1tzEbh6qkd6uzxPWHhZWL5ry7q8FRltcG?usp=sharing)
* [Caltech](https://drive.google.com/drive/folders/1tzEbh6qkd6uzxPWHhZWL5ry7q8FRltcG?usp=sharing)

For Foggycityscapes, we recommend adopting the identical file structure as that of Cityscapes and utilizing the same label scheme as applied in Cityscapes. Specifically, please organize it into the following format:
````
Foggycityscapes
  - images
    - train_02
    - train_01
    - train_005
    - val_02
    - val_01
    - val_005
  - labels
    - train_02
    - train_01
    - train_005
    - val_02
    - val_01
    - val_005
````

### Usage
<details open>
<summary>Conda environment</summary>

Clone repo and install [requirements.txt](https://github.com/caiyancheng/BFDA/blob/main/requirements.txt) in a
[**Python>=3.8.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/caiyancheng/BFDA.git  # clone
cd BFDA
pip install -r requirements.txt  # install
```

</details>

<details open>
<summary>Download the YOLOv5 pre-trained models</summary>

Due to the continuous iteration of the original YOLOv5 repo, the pre-trained weights used by the BFDA framework can be downloaded here:
[YOLOv5 pre-trained models](https://drive.google.com/drive/folders/1I5zM935VgVTQt7rT0adL_K5ajRt0fFDv?usp=sharing). Please place the downloaded weight file in the BFDA root directory.

</details>

<details open>
<summary>Set data path</summary>

After cloning this repository and downloading the pre-trained weights, please create a '[your_data_set].yaml' file in the './data' directory. Mimic the format of the other YAML files in this path.

</details>

<details open>
<summary>Set hyperparameters</summary>

Find "parser.add_argument" in each python file when you need to run the py file and set the internal hyperparameters. Hyperparameters in [hyp.scratch.yaml](https://github.com/caiyancheng/BFDA/blob/main/data/hyp.scratch.yaml) can also be modified.

- Note that BFDA's adversarial learning strategy is sensitive to hyperparameters, so it's recommended to run multiple iterations with the same set of hyperparameters.

</details>

<details open>
<summary>Train source domain weights (source)</summary>

Taking Cityscapes -> Caltech as an example, start by training YOLOv5 detection weights on the source domain, Cityscapes.

```bash
python train_city_tip.py # -- hyperparameters
```

</details>


### Citation
If you find this work helpful in your research, please cite.
````
@article{cai2023rethinking,
  title={Rethinking cross-domain pedestrian detection: a background-focused distribution alignment framework for instance-free one-stage detectors},
  author={Cai, Yancheng and Zhang, Bo and Li, Baopu and Chen, Tao and Yan, Hongliang and Zhang, Jingdong},
  journal={IEEE transactions on image processing},
  year={2023},
  publisher={IEEE}
}
````

### Acknowledgement
This work was supported in part by the National Natural Science Foundation of China under Grant 62071127 and Grant U1909207, in part by the Shanghai Natural Science Foundation
under Grant 23ZR1402900, and in part by the Zhejiang Laboratory under Project 2021KH0AB05.

We also greatly acknowledge the authors of _YOLOv5_ for their open-source codes. Visit the following links to access more contributions of them.
* [YOLOv5](https://github.com/ultralytics/yolov5)

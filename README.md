[![IEEE](https://img.shields.io/badge/IEEE-10231122-b31b1b.svg)](https://ieeexplore.ieee.org/document/10231122)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/caiyancheng/BFDA/pulls)

# T-IP-2023: Rethinking Cross-Domain Pedestrian Detection: A Background-Focused Distribution Alignment Framework for Instance-Free One-Stage Detectors

### Abstract
Cross-domain pedestrian detection aims to generalize pedestrian detectors from one label-rich domain to another label-scarce domain, which is crucial for various real-world applications. Most recent works focus on domain alignment to train domain-adaptive detectors either at the instance level or image level. From a practical point of view, one-stage detectors are faster. Therefore, we concentrate on designing a cross-domain algorithm for rapid one-stage detectors that lacks instance-level proposals and can only perform image-level feature alignment. However, pure image-level feature alignment causes the foreground-background misalignment issue to arise, i.e., the foreground features in the source domain image are falsely aligned with background features in the target domain image. To address this issue, we systematically analyze the importance of foreground and background in image-level cross-domain alignment, and learn that background plays a more critical role in image-level cross-domain alignment. Therefore, we focus on cross-domain background feature alignment while minimizing the influence of foreground features on the cross-domain alignment stage. This paper proposes a novel framework, namely, background-focused distribution alignment (BFDA), to train domain adaptive onestage pedestrian detectors. Specifically, BFDA first decouples the background features from the whole image feature maps and then aligns them via a novel long-short-range discriminator. Extensive experiments demonstrate that compared to mainstream domain adaptation technologies, BFDA significantly enhances cross-domain pedestrian detection performance for either one-stage or two-stage detectors. Moreover, by employing the efficient one-stage detector (YOLOv5), BFDA can reach 217.4 FPS (640×480 pixels) on NVIDIA Tesla V100 (7∼12 times the FPS of the existing frameworks), which is highly significant for practical applications.

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
We greatly acknowledge the authors of _YOLOv5_ for their open-source codes. Visit the following links to access more contributions of them.
* [YOLOv5](https://github.com/ultralytics/yolov5)

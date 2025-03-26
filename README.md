#  Learning Cross-Task Features With Mamba for Remote Sensing Image Multitask Prediction

In this repository, we implement RSMTMamba, an innovative MTL framework that integrates Mamba for multitask prediction in RS images. Our network simultaneously performs semantic segmentation, height estimation, and boundary detection within a unified architecture. The proposed architecture prioritizes the decoder, with a shared encoder for feature extraction. Specifically, a Mamba-based cross-task feature learning (MCFL) module is introduced to capture the interrelations among different tasks. 

The paper is here  [Learning Cross-Task Features With Mamba for Remote Sensing Image Multitask Prediction](https://ieeexplore.ieee.org/abstract/document/10879310).

The training and testing code can refer to [GeoSeg](https://github.com/WangLibo1995/GeoSeg).

## Install Mamba
```
pip install causal-conv1d>=1.4.0
pip install mamba-ssm
```

## Cite
If our code is helpful, please cite:

Y. Shen, L. Xiao, J. Chen, Q. Du and Q. Ye, "Learning Cross-Task Features With Mamba for Remote Sensing Image Multitask Prediction," IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1-16, 2025, Art no. 5612116, doi: 10.1109/TGRS.2025.3540573

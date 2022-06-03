# Crowd Counting via Segmentation Guided Attention Networks and Curriculum Loss
A Pytorch implementation of SGANet for crowd counting
## Abstract
   Automatic crowd behaviour analysis is an important task for intelligent transportation systems to enable effective flow control and dynamic route planning for varying road participants. Crowd counting is one of the keys to automatic crowd behaviour analysis.
   Crowd counting using deep convolutional neural networks (CNN) has achieved encouraging progress in recent years. Researchers have devoted much effort to the design of variant CNN architectures and most of them are based on the pre-trained VGG16 model. Due to the insufficient expressive capacity, the backbone network of VGG16 is usually followed by another cumbersome network specially designed for good counting performance. Although VGG models have been outperformed by Inception models in image classification tasks, the existing crowd counting networks built with Inception modules still only have a small number of layers with basic types of Inception modules. To fill in this gap, in this paper, we firstly benchmark the baseline Inception-v3 model on commonly used crowd counting datasets and achieve surprisingly good performance comparable with or better than most existing crowd counting models. Subsequently, we push the boundary of this disruptive work further by proposing a Segmentation Guided Attention Network (SGANet) with Inception-v3 as the backbone and a novel curriculum loss for crowd counting. We conduct thorough experiments to compare the performance of our SGANet with prior arts and the proposed model can achieve state-of-the-art performance with MAE of 57.6, 6.3 and 87.6 on ShanghaiTechA, ShanghaiTechB and UCF\_QNRF, respectively.

## How to use
1. Modify the data path and  parameter settings as needed
2. Use this command to train and test: python3 headCounting_shanghaitech_segLoss.py
## Reference
@article{wang22crowds,\
 author = {Wang, Q. and Breckon, T.P.}, \
 title = {Crowd Counting via Segmentation Guided Attention Networks and Curriculum Loss},\
 journal = {IEEE Trans. Intelligent Transportation Systems},\
 year = {2022},\
 month = {},\
 publisher = {IEEE},\
 keywords = {crowd counting, curriculum loss, inception-v3, segmentation guided attention networks, convolutional neural networks},\
 url = {https://breckon.org/toby/publications/papers/wang22crowds.pdf}, \
 doi = {10.1109/TITS.2021.3138896}, \
 arxiv = {https://arxiv.org/abs/1911.07990}, \
 note = {to appear},\
 category = {surveillance},\
}
## Contact
qian.wang173@hotmail.com

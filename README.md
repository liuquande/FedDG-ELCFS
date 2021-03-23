# FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space
by [Quande Liu](https://github.com/liuquande), [Cheng Chen](https://cchen-cc.github.io/), [Jing Qin](https://sn.polyu.edu.hk/en/people/academic_staff/index.html#harry.qin), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

This repository is for our CVPR 2021 paper '[FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space](https://arxiv.org/pdf/2103.06030.pdf)'. 

![](figure/cvpr21_feddg.png)

### Usage

1. Start with a demo for continuous frequency space interpolation among federated clicnets:
   ```shell
   python freq_space_interpolation_demo.py
   ```
<p align="center">
   <img src="figure/demo.png" width="600"/>
</p>

2. Prepare the dataset, and then extract the amplitude spectrum of samples in each local client with the function in ``dataset/prepare_dataset.py``:

3. Organize the data (saved sa npy) and amplitude spectrum of local clients as following structure:
   ``` 
     ├── dataset
        ├── client1
           ├── data_npy
               ├── sample1.npy, sample2.npy, xxxx
           ├── freq_amp_npy
               ├── amp_sample1.npy, amp_sample2.npy, xxxx
        ├── clientxxx
        ├── clientxxx
   ```
4. Train the federated learning model with ELCFS:
   ```shell
   python train_ELCFS.py
   ```
   
### Citation
If this repository is useful for your research, please consider citing:
```
@article{liu2021feddg,
  title={FedDG: Federated Domain Generalization on Medical Image Segmentation via Episodic Learning in Continuous Frequency Space},
  author={Liu, Quande and Chen, Cheng and Qin, Jing and Dou, Qi and Heng, Pheng-Ann},
  journal={The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

#### Acknowledgement
Some of the code is adapted from [SAML](https://github.com/liuquande/SAML) and [FDA](https://github.com/YanchaoYang/FDA). The dataset used in this paper is downloaded from [Prostate](https://liuquande.github.io/SAML/) and [Fundus](https://github.com/EmmaW8/Dofe).

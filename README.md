# MADAN

## Code and Dataset
code of Cross-regional oil palm tree counting and detection via multi-level attention domain adaptation network

We hope that our work can help other researchers who are interested in domain adaptative tree crown detection.

Our dataset can be downloaded from

[Google Drive](https://drive.google.com/drive/folders/1VHmx7LRPfKBkunKWxWQfZu9y3IZ0MuX3?usp=sharing)

[Baidu Wangpan](https://pan.baidu.com/s/1KROJNDmEJe3x97spm65k0A)  Access: eqn5

## Training & Testing

```
python madan.py path_of_source_dataset path_of_target_dataset id_of_GPU
```

for example:

```
python madan.py /data/zjp/ITCD/palm/dataset/0 /data/zjp/ITCD/palm/dataset/1 0
```

In the directory of the source or target dataset here, we both include all land cover types, such as palm, vegetation, bare land, and impervious.

Of course, if you need to use your definition of dataset, you can change the file data_loader.py

## Citation

If you use this code for your research, please consider citing:

```
@article{zheng2020cross,
  title={Cross-regional oil palm tree counting and detection via a multi-level attention domain adaptation network},
  author={Zheng, Juepeng and Fu, Haohuan and Li, Weijia and Wu, Wenzhao and Zhao, Yi and Dong, Runmin and Yu, Le},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={167},
  pages={154--177},
  year={2020},
  publisher={Elsevier}
}
```

Zheng, J., Fu, H., Li, W., Wu, W., Zhao, Y., Dong, R., & Yu, L. (2020). Cross-regional oil palm tree counting and detection via a multi-level attention domain adaptation network. ISPRS Journal of Photogrammetry and Remote Sensing, 167, 154-177.

## Contact

zjp19@mails.tsinghua.edu.cn


## Paper link
[Cross-regional oil palm tree counting and detection via a multi-level attention domain adaptation network](https://doi.org/10.1016/j.isprsjprs.2020.07.002)

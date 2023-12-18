# VIAT: Viewpoint-Invariant Adversarial Training
This repository contains the official implementation of VIAT and GMVFool for the paper ["Towards Viewpoint-Invariant Visual Recognition via Adversarial Training"](https://arxiv.org/pdf/2307.10235.pdf) (ICCV2023) 

and ["Improving viewpoint robustness for visual recognition via adversarial training"](https://scholar.google.cz/citations?view_op=view_citation&hl=zh-CN&user=1pggtuUAAAAJ&citation_for_view=1pggtuUAAAAJ:2osOgNQ5qMEC) (Extended Version)

By [Shouwei Ruan](https://heathcliff-saku.github.io/), [Yinpeng Dong](https://ml.cs.tsinghua.edu.cn/~yinpeng/), [Hang Su](https://www.suhangss.me/), Jianteng Peng, Ning Chen, [Xingxing Wei](https://sites.google.com/site/xingxingwei1988/)

![fig1](asset/framework.png)

## ⚙️ 1.Prerequisites
Ensure that you have these environments:
- Pytorch (1.11.0)
- torchvision (0.13.1)
- timm (0.5.4)
- pytorch-lighting(1.6.5)
  
1. For the complete running environment of VIAT, please refer to `./requirements.txt`.
2. (🔥**Extremely important**) Please ensure to follow the [ngp_pl(Instant-NGP based on pytorch-lighting)](https://github.com/kwea123/ngp_pl) to install the relevant environment for Instant-NGP, since VIAT and GMVFool depend on Instant-NGP for viewpoint rendering.

## 💾 2.Datasets
The released data consists of two parts: 
1. `IM3D`: A multi-view dataset used for training, containing 100 hemispherical sampled viewpoint renderings of 1000 virtual 3D objects.
- [IM3D download link (Google Drive)]()

2. `ImageNet-V+`, A larger adversarial viewpoint benchmark, including 100K adversarial viewpoint samples captured by GMVFool on IM3D.
- [IM3D download link (Google Drive)](https://drive.google.com/file/d/1oxrWl4mRa_mEr-ByCMhyRWaQG8Wribo7/view)



## 💾 3.Training NeRF for IM3D Objects






... The code is currently being organized, thank you for your patience.


## Citation
If you find our methods useful or use the IM3D and imagenet-v+ dataset, please consider citing:

```
@inproceedings{ruan2023towards,
  title={Towards Viewpoint-Invariant Visual Recognition via Adversarial Training},
  author={Ruan, Shouwei and Dong, Yinpeng and Su, Hang and Peng, Jianteng and Chen, Ning and Wei, Xingxing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={4709--4719},
  year={2023}
}
```
```
@article{ruan2023improving,
  title={Improving viewpoint robustness for visual recognition via adversarial training},
  author={Ruan, Shouwei and Dong, Yinpeng and Su, Hang and Peng, Jianteng and Chen, Ning and Wei, Xingxing},
  journal={arXiv preprint arXiv:2307.11528},
  year={2023}
}
```
This project uses Unofficial implementation of Instant-NGP (ngp_pl), Thanks to @kwea123:

```
@misc{queianchen_nerf,
  author={Quei-An, Chen},
  title={Nerf_pl: a pytorch-lightning implementation of NeRF},
  url={https://github.com/kwea123/nerf_pl/},
  year={2020},
}
```

If you are interested in viewpoint robustness, welcome to check our previous work: [ViewFool: Evaluating the Robustness of Visual Recognition to Adversarial Viewpoints (NIPS2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/eee7ae5cf0c4356c2aeca400771791aa-Paper-Conference.pdf) [[project]](https://github.com/Heathcliff-saku/ViewFool_).



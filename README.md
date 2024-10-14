<p align="center" />
<h1 align="center">Binocular-Guided 3D Gaussian Splatting with ViewConsistency for Sparse View Synthesis </h1>

<p align="center">
    <strong>Liang Han</strong>
    ·
    <strong>Junshen Zhou</strong>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>
</p>
<h2 align="center">NeurIPS 2024</h2>
<h3 align="center"><a href="#">Paper</a> | <a href="#">Project Page</a></h3>
<div align="center"></div>
<p align="center">
    <img src="assets/pipeline.png" width="780" />
</p>

We leverage dense initialization for achieving Gaussian locations, and optimize the locations and Gaussian attributes with three constraints or strategies:
<ul>
<li> Binocular Stereo Consistency Loss. We construct a binocular view pair by translating an input view with camera positions, where we constrain on the view consistency of binocular view pairs in a self-supervised manner.</li>
<li> Opacity Penalty Strategy is designed to decay the Gaussian opacity during training for regularizing them. </li>
<li> The commonly-used Color Reconstruction Loss. </li>
</ul>

# Installation

# Dataset

# Training

# Evaluation


# Citation
If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{han2024binocular,
    title = {Binocular-Guided 3D Gaussian Splatting with View Consistency for Sparse View Synthesis},
    author = {Han, Liang and Zhou, Junsheng and Liu, Yu-Shen and Han, Zhizhong},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year = {2024}
}
```

# Acknowledgement
This project is built upon [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) with [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) and a modified [diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization). 
The scripts for generating videos are borrowed from [DNGaussian](https://fictionarry.github.io/DNGaussian) and the scripts for dense matching are from [PDCNet+](https://prunetruong.com/pdcnet+). Thanks for these great projects.
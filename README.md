<h1 align="center"> Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail (CVPR 2025) </h1> 

<h3 align="center"> Evaluation code released - training code coming soon </h3>

<br>

:rotating_light: This repository will contain download links to our code, and trained deep stereo models of our work  "**Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail**",  [CVPR 2025](http://arxiv.org/abs/2412.04472)
 
by [Luca Bartolomei](https://bartn8.github.io/)<sup>1,2</sup>, [Fabio Tosi](https://fabiotosi92.github.io/)<sup>2</sup>, [Matteo Poggi](https://mattpoggi.github.io/)<sup>1,2</sup>, and [Stefano Mattoccia](https://github.com/stefano-mattoccia)<sup>1,2</sup>

Advanced Research Center on Electronic System (ARCES)<sup>1</sup>
University of Bologna<sup>2</sup>

<div class="alert alert-info">

<h2 align="center"> 

 Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail (CVPR 2025)<br>

 [Project Page](https://stereoanywhere.github.io/) | [Paper](http://arxiv.org/abs/2412.04472) 
</h2>

<img src="./images/teaser.png" alt="Alt text" style="width: 800px;" title="architecture">
<p style="text-align: justify;"><strong>Stereo Anywhere: Combining Monocular and Stereo Strenghts for Robust Depth Estimation.</strong> Our model achieves accurate results on standard conditions (on Middlebury), while effectively handling non-Lambertian surfaces where stereo networks fail (on Booster) and perspective illusions that deceive monocular depth foundation models (on MonoTrap, our novel dataset).</p>

**Note**: 🚧 Kindly note that this repository is currently in the development phase. We are actively working to add and refine features and documentation. We apologize for any inconvenience caused by incomplete or missing elements and appreciate your patience as we work towards completion.

## :bookmark_tabs: Table of Contents

- [:bookmark\_tabs: Table of Contents](#bookmark_tabs-table-of-contents)
- [:clapper: Introduction](#clapper-introduction)
- [:inbox\_tray: Pretrained Models](#inbox_tray-pretrained-models)
- [:memo: Code](#memo-code)
- [:floppy_disk: Datasets](#floppy_disk-datasets)
- [:train2: Training](#train2-training)
- [:rocket: Test](#rocket-test)
- [:art: Qualitative Results](#art-qualitative-results)
- [:envelope: Contacts](#envelope-contacts)
- [:pray: Acknowledgements](#pray-acknowledgements)

</div>

## :clapper: Introduction

We introduce Stereo Anywhere, a novel stereo-matching framework that combines geometric constraints with robust priors from monocular depth Vision Foundation Models (VFMs). By elegantly coupling these complementary worlds through a dual-branch architecture, we seamlessly integrate stereo matching with learned contextual cues. Following this design, our framework introduces novel cost volume fusion mechanisms that effectively handle critical challenges such as textureless regions, occlusions, and non-Lambertian surfaces. Through our novel optical illusion dataset, MonoTrap, and extensive evaluation across multiple benchmarks, we demonstrate that our synthetic-only trained model achieves state-of-the-art results in zero-shot generalization, significantly outperforming existing solutions while showing remarkable robustness to challenging cases such as mirrors and transparencies.

**Contributions:** 

* A novel deep stereo architecture leveraging monocular depth VFMs to achieve strong generalization capabilities and robustness to challenging conditions.

* Novel data augmentation strategies designed to enhance the robustness of our model to textureless regions and non-Lambertian surfaces.

* A challenging dataset with optical illusion, which is particularly challenging for monocular depth with VFMs.

* Extensive experiments showing Stereo Anywhere's superior generalization and robustness to conditions critical for either stereo or monocular approaches.


:fountain_pen: If you find this code useful in your research, please cite:

```bibtex
@article{bartolomei2024stereo,
  title={Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even Where Either Stereo or Mono Fail},
  author={Bartolomei, Luca and Tosi, Fabio and Poggi, Matteo and Mattoccia, Stefano},
  journal={arXiv preprint arXiv:2412.04472},
  year={2024},
}
```

## :inbox_tray: Pretrained Models

Here, you will be able to download the weights of our proposal trained on [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html).

You can download our pretrained models [here](https://drive.google.com/drive/folders/1uQqNJo2iWoPtXlSsv2koAt2OPYHpuh1x?usp=sharing).

## :memo: Code

Details about training and testing scripts will be released soon.


## :floppy_disk: Datasets

We used [Sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) dataset for training and eight datasets for evaluation.

Specifically, we evaluate our proposal and competitors using:
- 5 indoor/outdoor datasets: [Middlebury 2014](https://vision.middlebury.edu/stereo/data/scenes2014/), [Middlebury 2021](https://vision.middlebury.edu/stereo/data/scenes2021/), [ETH3D](https://www.eth3d.net/datasets), [KITTI 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo);
- two datasets **containing non-Lambertian** surfaces: [Booster](https://cvlab-unibo.github.io/booster-web/) and [LayeredFlow](https://layeredflow.cs.princeton.edu/);
- and finally with **MonoTrap** our novel stereo dataset specifically designed to challenge monocular depth estimation.

Details about datasets will be released soon.

## :train2: Training

We will provide futher information to train Stereo Anywhere soon.

## :rocket: Test

Evaluation command example:

```bash
python test.py --datapath <DATAPATH> --dataset <DATASET> \ 
--stereomodel stereoanywhere --loadstereomodel <STEREO_MODEL_PATH> \
--monomodel DAv2 --loadmonomodel <MONO_MODEL_PATH> \
--iscale <ISCALE> --oscale <OSCALE> --normalize --iters 32 \
--vol_n_masks 8 --n_additional_hourglass 0 \
--use_aggregate_mono_vol --vol_downsample 0 \
--mirror_conf_th 0.98  --use_truncate_vol --mirror_attenuation 0.9 
```

We will provide futher information to test Stereo Anywhere soon.

## :art: Qualitative Results

In this section, we present illustrative examples that demonstrate the effectiveness of our proposal.

<br>

<p float="left">
  <img src="./images/qualitative1.png" width="800" />
</p>
 
**Qualitative Results -- Zero-Shot Generalization.** Predictions by state-of-the-art models and Stereo Anywhere. In particular the first row shows an extremely challenging case for SceneFlow-trained models, where Stereo Anywhere achieves accurate disparity maps thanks to VFM priors.
 
<br>

<p float="left">
  <img src="./images/qualitative2.png" width="800" />
</p>

**Qualitative results -- Zero-Shot non-Lambertian Generalization.** Predictions by state-of-the-art models and Stereo Anywhere. Our proposal is the only stereo model correctly perceiving the mirror and transparent railing.

<br>

<p float="left">
  <img src="./images/qualitative3.png" width="800" />
</p>

**Qualitative results -- MonoTrap.** The figure shows three samples where Depth Anything v2 fails while Stereo Anywhere does not.

## :envelope: Contacts

For questions, please send an email to luca.bartolomei5@unibo.it

## :pray: Acknowledgements

We would like to extend our sincere appreciation to the authors of the following projects for making their code available, which we have utilized in our work:

- We would like to thank the authors of [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) for providing their code, which has been inspirational for our stereo matching architecture.
- We would like to thank also the authors of [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) for providing their incredible monocular depth estimation network that fuels our proposal Stereo Anywhere.


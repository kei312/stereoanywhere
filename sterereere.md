This paper has been accepted for publication at the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Nashville, 2025. ©IEEE
Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching
Even Where Either Stereo or Mono Fail
Luca Bartolomei∗,† Fabio Tosi† Matteo Poggi∗,† Stefano Mattoccia∗,†
∗Advanced Research Center on Electronic System (ARCES)
†Department of Computer Science and Engineering (DISI)
University of Bologna, Italy
https://stereoanywhere.github.io/
RGB Depth Anything v2 [121] RAFT-Stereo [55] Stereo Anywhere (Ours)
Middlebury
✓ ✓ ✓
Booster
✓ ✗ ✓
MonoTrap
Figure 1. Stereo Anywhere: Combining Monocular and Stereo Strenghts for Robust Depth Estimation.
✗ ✓
Our model achieves accurate
✓
results on standard conditions (on Middlebury [86]), while effectively handling non-Lambertian surfaces where stereo networks fail (on
Booster [127]) and perspective illusions that deceive monocular depth foundation models (on MonoTrap, our novel dataset).
Abstract
We introduce Stereo Anywhere, a novel stereo-matching
framework that combines geometric constraints with robust priors from monocular depth Vision Foundation Models (VFMs). By elegantly coupling these complementary
worlds through a dual-branch architecture, we seamlessly
integrate stereo matching with learned contextual cues. Following this design, our framework introduces novel cost
volume fusion mechanisms that effectively handle critical
challenges such as textureless regions, occlusions, and nonLambertian surfaces. Through our novel optical illusion
dataset, MonoTrap, and extensive evaluation across multiple benchmarks, we demonstrate that our synthetic-only
trained model achieves state-of-the-art results in zero-shot
generalization, significantly outperforming existing solutions while showing remarkable robustness to challenging
cases such as mirrors and transparencies.
1. Introduction
Stereo is a fundamental task that computes depth from a
synchronized, rectified image pair by finding pixel correspondences to measure their horizontal offset (disparity).
Due to its effectiveness and minimal hardware requirements, stereo has become prevalent in numerous applications, from autonomous navigation to augmented reality.
Although in principle single-image depth estimation [3]
requires an even simpler acquisition setup, its ill-posed nature leads to scale ambiguity and perspective illusion issues that stereo methods inherently overcome through wellestablished geometric multi-view constraints.
However, despite significant advances through deep
learning [47, 72], stereo models still face two main challenges: (i) limited generalization across different scenar1
arXiv:2412.04472v2 [cs.CV] 7 May 2025
ios, and (ii) critical conditions that hinder matching or
proper depth triangulation. Regarding (i), despite the initial success of synthetic datasets in enabling deep learning for stereo, their limited variety and simplified nature
poorly reflect real-world complexity, and the scarcity of real
training data further hinders the ability to handle heterogeneous scenarios. As for (ii), large textureless regions common in indoor environments make pixel matching highly
ambiguous, while occlusions and non-Lambertian surfaces
[76, 115, 127] violate the fundamental assumptions linking
pixel correspondences to 3D geometry.
We argue that both challenges are rooted in the underlying limitations of stereo training data. Indeed, while data
has scaled up to millions - or even billions - for several
computer vision tasks, stereo datasets are still constrained
in quantity and variety. This is particularly evident for nonLambertian surfaces, which are severely underrepresented
in existing datasets as their material properties prevent reliable depth measurements from active sensors (e.g. LiDAR).
In contrast, single-image depth estimation has recently
witnessed a significant scale-up in data availability, reaching the order of millions of samples and enabling the emergence of Vision Foundation Models (VFMs) [22, 43, 120,
121]. Such data abundance has influenced these models
in different ways, either through direct training on largescale depth datasets [120, 121] or indirectly by leveraging networks pre-trained on billions of images for diverse
tasks [22, 43]. Since these models rely on contextual
cues for depth estimation, they show better capability in
handling textureless regions and non-Lambertian materials
[75, 81, 128, 129] while being inherently immune to occlusions. Modern graphics engines have further accelerated this progress, enabling rapid generation of high-quality
synthetic data with dense depth annotations. However,
although synthetic datasets featuring non-Lambertian surfaces like HyperSim [81] have proven effective for monocular depth estimation [75, 128, 129], this data abundance has
not translated to stereo. Despite efforts in generating stereo
pairs via novel view synthesis [24, 54, 104], available data
remains insufficient for robust stereo matching.
In this paper, rather than focusing on costly real-world
data collection or generating additional synthetic datasets,
we propose to bridge this gap by leveraging existing VFMs
for single-view depth estimation. To this end, we develop
a novel dual-branch deep architecture that combines stereo
matching principles with monocular depth cues. Specifically, while one branch of the proposed network constructs
a cost volume from learned stereo image features, the other
branch processes depth predictions from the VFM on both
left and right images to build a second cost volume that
incorporates depth priors to guide the disparity estimation
process. These complementary signals are then iteratively
combined [55], along with novel augmentation strategies
applied to both cost volumes, to predict the final disparity map. Through this design, our network achieves robust performance on challenging cases like textureless regions, occlusions, and non-Lambertian surfaces, while requiring minimal synthetic stereo data. Importantly, while
leveraging monocular cues, our approach preserves stereo
matching geometric guarantees, effectively handling scenarios where monocular depth estimation typically fails,
such as in the presence of perspective illusions. We validate
this through our novel dataset of optical illusions, comprising 26 scenes with ground-truth depth maps.
We dub our framework Stereo Anywhere, highlighting its
ability to overcome the individual limitations of stereo and
monocular approaches, as depicted in Fig. 1. To summarize, our main contributions are:
• A novel deep stereo architecture leveraging monocular
depth VFMs to achieve strong generalization capabilities
and robustness to challenging conditions.
• Novel data augmentation strategies designed to enhance
the robustness of our model to textureless regions and
non-Lambertian surfaces.
• A challenging dataset with optical illusion, which is particularly challenging for monocular depth with VFMs.
• Extensive experiments showing Stereo Anywhere’s superior generalization and robustness to conditions critical
for either stereo or monocular approaches.
2. Related Works
We briefly review the literature relevant to our work.
Deep Stereo Matching. In the last decade, stereo matching has transitioned from classical hand-crafted algorithms
[85] to deep learning solutions, leading to unprecedented
accuracy in depth estimation. Early deep learning efforts
focused on replacing individual components of the conventional pipeline [88, 96, 105, 130, 131]. Since DispNetC
[61], end-to-end architectures have evolved into 2D [53, 92,
125, 125] and 3D [4, 8, 9, 32, 44, 90, 91, 119, 132, 134]
approaches, processing cost volumes through correlation
layers or 3D convolutions respectively. More recent advances, thoroughly reviewed in [47, 72, 107], include recurrent architectures for stereo matching [13, 27, 40, 50,
55, 110, 116, 140] inspired by RAFT [99], Transformerbased solutions [31, 52, 59, 97, 113, 117, 138] for capturing
long-range dependencies, and fully data-driven MRF models [28]. Among them, some methods specifically address
temporal consistency in stereo videos [41, 42, 133, 137].
Domain generalization remains a major challenge, with various approaches proposed including domain-invariant feature learning [17, 56, 80, 93, 135], hand-crafted matching costs [7, 15], integration of additional geometric cues
[2, 66, 105], and exploitation of sparse depth measurements from active sensors [5, 49, 69]. In parallel, selfsupervised approaches [25, 57] have emerged as effec2
Correlation
 Volume
from normals
Aggregated
Correlation
 Volume
from normals
Correlation
 Volume
Truncate
Function
Truncated
Correlation
Volume
Estimated 3D Hourglass
Normal Maps
Context
 Backbone
Feature extraction
 Backbone
Stereo Pair
Monocular
Depth Estimations
(MDEs)
Differentiable
Scaler
L
L
Scaled MDE
L
L
L
L
Final
Disparity
Correlation Pyramids
Correlation Pyramids
from normals
(1) (2)
(3) (4)
(6)
(5) (7)
3.1 Features
Extraction
3.2 Correlation Pyramids
Building
3.3 Iterative Disparity
Estimation
Figure 2. Stereo Anywhere Architecture. Given a stereo pair, (1) a pre-trained backbone is used to extract features and then build a
correlation volume. Such a volume is then truncated (2) to reject matching costs computed for disparity hypotheses being behind nonLambertian surfaces – glasses and mirrors. On a parallel branch, the two images are processed by a monocular VFM to obtain two depth
maps (3): these are used to build a second correlation volume from retrieved normals (4). This volume is then aggregated through a 3D
CNN to predict a new disparity map, used to align the original monocular depth to metric scale through a differentiable scaling module (5)
for it. In parallel, the monocular depth map from left images is processed by another backbone (6) to extract context features. Finally, the
two volumes and the context features from monocular depth guide the iterative disparity prediction (7).
tive alternatives to supervised learning, even using pseudolabels from traditional algorithms [1, 100] or deploying neural radiance fields [104]. Despite the numerous attempts to
improve specific aspects through the aforementioned techniques, recent architectures achieve remarkable generalization by combining their architectural advances with the increasing availability of diverse training data, while online
adaptation techniques enable further improvements during deployment through self-supervised learning [45, 67,
71, 101]. However, although progress on challenges like
over-smoothing [103, 118] and visually imbalanced stereo
[2, 11, 58, 105], handling non-Lambertian surfaces remains particularly challenging due to limited annotated data
and complex appearance, with rare works like Depth4ToM
[18] specifically addressing this through semantic guidance.
Among all the aforementioned approaches, there have been
limited attempts to integrate stereo with monocular cues
[1, 12, 112], mostly in self-supervised settings or through
loose coupling between modalities.
Monocular Depth Estimation. Parallel to developments in stereo matching, single-image depth estimation
has evolved from hand-crafted features [82] to deep learning methods [10, 21, 48, 73, 108], with self-supervised approaches [25, 26, 60, 68, 111, 139, 141] reframing the task
as an image reconstruction problem. This led to multi-task
approaches incorporating flow [79, 102, 124, 142] and semantics [29, 126], alongside advances in uncertainty estimation [34, 70] and dynamic object handling [46, 63, 98].
Affine-invariant models [20, 77, 78, 109, 122] marked a
breakthrough in cross-domain generalization, pioneered by
MiDaS [78] and followed by works like DPT [77] and,
more recently, the Depth Anything series [120]. These
approaches used different data sources, from internet photos [51, 94, 95, 122] to car sensors [23, 62] and RGB-D devices [16, 64], representing the first generation of VFMs for
monocular depth estimation. Recent works have focused
on metric depth estimation through camera parameter integration [30, 35, 123], diffusion models [19, 22, 33, 38,
43, 83, 84], and temporal consistency [36, 89]. Moreover,
material-aware methods [18], diffusion models [106], and
large-scale synthetic datasets have enabled robust monocular depth estimation for non-Lambertian surfaces [121].
Stereo methods, however, still struggle with these surfaces
due to limited real-world and synthetic annotated data, affecting generalization. We address this by integrating robust
monocular VFMs into a stereo architecture.
Concurrent Works. Finally, we mention some solutions
for stereo [14, 39, 114] and for multi-view stereo [37], developed in parallel with ours and sharing similar rationale.
3. Method Overview
Given a rectified stereo pair IL, IR ∈ R
3×H×W , we first
obtain monocular depth estimates (MDEs) ML,MR ∈
R
1×H×W using a generic VFM ϕM for monocular depth
estimation. We aim to estimate a disparity map D =
ϕS(IL, IR,ML,MR), incorporating VFM priors to provide accurate results even under challenging conditions,
such as texture-less areas, occlusions, and non-Lambertian
surfaces. At the same time, our stereo network ϕS is designed to avoid depth estimation errors that could arise from
relying solely on contextual cues, which can be ambiguous,
like in the presence of visual illusions.
Following recent advances in iterative models [55],
Stereo Anywhere comprises three main stages, as shown
3
in Fig. 2: I) Feature Extraction, II) Correlation Pyramids
Building, and III) Iterative Disparity Estimation.
3.1. Feature Extraction
Two distinct types of features are extracted [55]: image
features and context features – (1) and (6) in Fig. 2. The
image features are obtained through a feature encoder processing the stereo pair, yielding feature maps FL, FR ∈
R
D× H
4 × W
4 , which are used to build a stereo correlation volume at 1
4
of the original input resolution. These encoders
are initialized with pre-trained weights [55] and the image
encoder is kept frozen during training. For context features,
we employ a context encoder with identical architecture to
the feature encoder, but processing the monocular depth estimate aligned with the reference image ML – (3) in Fig. 2
– instead of IL to capture strong geometry priors. Accordingly, during training the context encoder is optimized to
extract meaningful features from these depth maps.
3.2. Correlation Pyramids Building
As a standard practice in stereo matching, the cost volume
is the data structure encoding the similarity between pixels across two images. Accordingly, our model utilizes cost
volumes—specifically Correlation Pyramids [55]—but in a
novel manner. Indeed, Stereo Anywhere constructs two correlation pyramids: a stereo correlation volume derived from
IL, IR to encode image similarities, and a monocular correlation volume from ML,MR to encode geometric similarities—(2) and (4) in Fig. 2. Unlike the former, the latter
remains unaffected by non-Lambertian surfaces, assuming
a robust ϕM.
Stereo Correlation Volume. Given FL, FR, we construct a 3D correlation volume VS using dot product between feature maps:
 (\mathbf {V}_S)_{ijk} = \sum _{h} (\mathbf {F}_L)_{hij} \cdot (\mathbf {F}_R)_{hik}, \ \mathbf {V}_S \in \mathbb {R}^{\frac {H}{4} \times \frac {W}{4} \times \frac {W}{4}} \label {eq:dot_corr} (1)
Monocular Correlation Volume. Given ML,MR, we
downsample them to 1/4, compute their normals ∇L, ∇R,
and construct a 3D correlation volume VM using dot product between normal maps:
 (\mathbf {V}_M)_{ijk} = \sum _{h} (\nabla _L)_{hij} \cdot (\nabla _R)_{hik}, \ \mathbf {V}_M \in \mathbb {R}^{\frac {H}{4} \times \frac {W}{4} \times \frac {W}{4}} \label {eq:dot_corr_mono}
(2)
Given the absence of texture in ∇L and ∇R, the resulting monocular volume VM will be less informative. To
alleviate this problem we segment VM using the relative
depth priors from ML and MR: to do so, we generate
left and right segmentation masks ML ∈ {0, 1}
H
4 × W
4 ×1
,
MR ∈ {0, 1}
H
4 ×1× W
4 . We refer the reader to the supplementary material for a detailed description. Given the
segmentation masks, we can generate masked volumes as:
 ({\mathbf {V}_M}^n)_{ijk} = ({\mathcal {M}_L}^n)_{ij} \cdot ({\mathcal {M}_R}^n)_{ik} \cdot (\mathbf {V}_M)_{ijk} \label {eq:vol_masking} (3)
Next, we insert a 3D Convolutional Regularization module ϕA to aggregate VM
n
, resulting in V′M =
ϕA(VM
1
, . . . , VM
N ,ML,MR), with N = 8. The architecture of ϕA follows the one in [116], with a simple permutation to match the structure of the correlation volumes. We
propose an adapted version of CoEx [4] correlation volume
excitation that exploits both views. The resulting feature
volumes V′M ∈ R
F × H
4 × W
4 × W
4 are fed to two different
shallow 3D conv layers ϕD and ϕC to obtain two aggregated volumes VD
M = ϕD(V′M) and VC
M = ϕC (V′M)
with VD
M, VC
M ∈ R
H
4 × W
4 × W
4 .
Differentiable Monocular Scaling. Volume VD
M will
be used not only as a monocular guide for the iterative refinement unit but also to estimate the coarse disparity maps
Dˆ L Dˆ R, while VC
M is used to estimate confidence maps Cˆ L
Cˆ R. These maps are then used to scale both ML and MR
– (5) in Fig. 2. To estimate left disparity from a correlation
volume, we first perform a softargmax on the last W dimension of VD
M to extract the correlated pixel x-coordinate.
Then, given the relationship between left disparity and correlation dL = jL − jR, we obtain a coarse disparity map
Dˆ L:
 (\hat {\mathbf {D}}_L)_{ij} = j - \left (\text {softargmax}_L(\mathbf {V}^D_M)\right )_{ij} \label {eq:softmax_left} (4)
Similarly, we estimate Dˆ R from VD
M. We refer the reader
to the supplementary for details. We also estimate a pair of
confidence maps Cˆ L, Cˆ R ∈ [0, 1]H×W to classify outliers
and perform robust scaling. Inspired by information entropy, we measure the chaos within correlation curves: clear
monomodal-like cost curves—those with low entropy—are
reliable, while chaotic curves with high entropy indicate uncertainty. To estimate the left confidence map, we perform
a softmax operation on the last W dimension of VC
M, then
Cˆ L is obtained as follows:
 (\hat {\mathbf {C}}_L)_{ij} = 1 + \frac {\sum _{d}^{\frac {W}{4}} \frac {e^{(\mathbf {V}^C_M)_{ijd}}}{\sum _{f}^{\frac {W}{4}} e^{(\mathbf {V}^C_M)_{ijf}}} \cdot \log _2 \left ( \frac {e^{(\mathbf {V}^C_M)_{ijd}}}{\sum _{f}^{\frac {W}{4}} e^{(\mathbf {V}^C_M)_{ijf}}} \right )}{\log _2(\frac {W}{4})} \label {eq:confidence_left}
(5)
In the same way, we estimate Cˆ R. To further reduce outliers, we mask out occluded pixels from Cˆ L and Cˆ R using
a SoftLRC operator – see the supplementary material for
details. Finally, we estimate the scale sˆ and shift tˆ using a
differentiable weighted least-square approach:
 \min _{\hat {s}, \hat {t}} \sum _{}^{L,R} \left \lVert \sqrt {\hat {\mathbf {C}}}\odot \left [\left (\hat {s}\mathbf {M} + \hat {t}\right ) - \hat {\mathbf {D}} \right ] \right \rVert _F \label {eq:scale_shift} (6)
where ∥·∥F denotes the Frobenius norm. Using the scaling
coefficients, we obtain two disparity maps Mˆ L, Mˆ R:
 \hat {\mathbf {M}}_L = \hat {s}\mathbf {M}_L + \hat {t},\ \hat {\mathbf {M}}_R = \hat {s}\mathbf {M}_R + \hat {t} \label {eq:scaling_op} (7)
4
Image Ground-Truth Depth Anything v2 [121] Image Ground-Truth Depth Anything v2 [121]
Figure 3. Samples from MonoTrap Dataset. We report two scenes featured in our dataset, showing the left image, the ground-truth depth,
and the predictions by Depth Anything v2 [121], highlighting how it fails in the presence of visual illusions.
It is crucial to optimize both left and right scaling jointly to
obtain consistency between Mˆ L and Mˆ R.
Volume Augmentations. Unfortunately, Stereo Anywhere cannot properly learn when to choose stereo or mono
information from [61] alone. Hence, we propose three volume augmentations and a monocular augmentation to overcome this issue: 1) Volume Rolling: we randomly apply
a rolling operation to the last W dimension of VD
M or
VS; 2) Volume Noising: we apply random noise sampled
from the interval [0, 1) using a uniform distribution; 3) Volume Zeroing: we apply a Gaussian-like curve with the peak
where disparity equals zero. Furthermore, we randomly
substitute the monocular depth with ground truth normalized between [0, 1] as an additional augmentation. We apply
only one volume augmentation to VD
M or VS and only for
a section of the volume, randomly selecting an Mn
L mask.
Volume Truncation. To further help Stereo Anywhere
to handle mirror surfaces, we introduce a hand-crafted volume truncation operation on VS. Firstly, we extract left
confidence CM = softLRCL(Mˆ L,Mˆ R) to classify reliable monocular predictions. Then, we create a truncate mask T ∈ [0, 1] H
4 × W
4 using the following logic
condition: (T)ij =
h(Mˆ L)ij > (Dˆ L)ij
∧ (CM)ij i
∨
h
(CM)ij ∧ ¬(Cˆ L)ij i
. We implement this logic using
fuzzy operators (more details in the supplementary material). The rationale is that stereo predicts farther depths
on mirror surfaces: the mirror is perceived as a window into
a new environment, specular to the real one. Finally, for
values of T > Tm = 0.98, we truncate VS using a sigmoid
curve centered at the correlation value predicted by Mˆ L –
i.e., the real disparity of mirror surfaces – preserving only
the stereo correlation curve not “piercing” mirrors.
3.3. Iterative Disparity Estimation
We aim to estimate a series of refined disparity maps {D1 =
Mˆ L, D2
, . . . Dl
, . . . } exploiting the guidance from both
stereo and mono branches. Starting from the Multi-GRU
update operator by [55], we introduce a second lookup operator that extracts correlation features GM from the additional volume VD
M – (7) in Fig. 2. The two sets of correlation features from GS and GM are processed by the same
two-layer encoder and concatenated with features derived
from the current disparity estimation Dl
. This concatenation is further processed by a 2D conv layer, and then by
the ConvGRU operator. We inherit the convex upsampling
module [55] to upsample final disparity to full resolution.
3.4. Training Supervision
We supervise the iterative module using the well-known L1
loss with exponentially increasing weights [55], then Dˆ L,
Dˆ R, Mˆ L and Mˆ R using the L1 loss, finally Cˆ L and Cˆ R
using the Binary Cross Entropy loss. We invite the reader
to read the supplementary material for additional details.
4. The MonoTrap Dataset
Monocular depth estimation is known for possibly failing
in the presence of perspective illusions. The reader may
wonder how Stereo Anywhere would behave in such cases:
would it blindly trust the monocular VFM or rely on the
stereo geometric principles to maintain robustness?
To answer these questions, we introduce MonoTrap,
a novel stereo dataset specifically designed to challenge
monocular depth estimation. Our dataset comprises 26
scenes featuring perspective illusions, captured with a calibrated stereo setup and annotated with ground-truth depth
from an Intel Realsense L515 LiDAR. The scenes contain
carefully designed planar patterns that create visual illusions, such as apparent holes in walls or floors and simulated transparent surfaces that reveal content behind them.
Figure 3 shows examples from our dataset that illustrate
how these visual illusions easily fool monocular methods.
5. Experiments
We describe our implementation details, datasets, and evaluation protocols, followed by experiments. We also refer
the reader to the supplementary material for more results.
5.1. Implementation and Experimental Settings
We implement Stereo Anywhere using PyTorch, starting
from RAFT-Stereo codebase [55]. We use Depth Anything
v2 [121] as the VFM fueling our model, using the Large
weights provided by the authors, trained on ground-truth
labels from the HyperSim synthetic dataset [81] only.
Starting from the Sceneflow RAFT-Stereo checkpoint,
we train Stereo Anywhere on a single A100 GPU for 3
epochs, with learning rate 1e-4 and AdamW optimizer, on
5
Booster (Q) Middlebury 2014 (H)
Experiment bad Avg. bad > 2 Avg.
> 2 > 4 > 6 > 8 (px) All Noc Occ (px)
(A) Baseline [55] 17.84 13.06 10.76 9.24 3.59 11.15 8.06 29.06 1.55
(B) (A) + Monocular Context w/o re-train 15.85 10.98 8.89 7.69 3.05 14.96 11.70 34.38 2.82
(C) (A) + Monocular Context w/ re-train 14.94 10.40 8.61 7.63 3.03 9.62 6.98 25.39 1.13
(D) (C) + Normals Correlation Volume / Scaled Depth 11.33 6.88 5.32 4.59 1.87 7.67 5.24 21.51 0.96
(E) (D) + Volume augmentation / truncation 9.01 5.40 4.12 3.34 1.21 6.96 4.75 20.34 0.94
Table 1. Ablation Studies. We measure the impact of different design strategies. Networks trained on SceneFlow [61].
Middlebury 2014 (H) Middlebury 2021 ETH3D KITTI 2012 KITTI 2015
Model bad > 2 Avg. bad > 2 Avg. bad > 1 Avg. bad > 3 Avg. bad > 3 Avg.
All Noc Occ (px) All Noc Occ (px) All Noc Occ (px) All Noc Occ (px) All Noc Occ (px)
RAFT-Stereo [55] 11.15 8.06 29.06 1.55 12.05 9.38 37.89 1.81 2.59 2.24 8.78 0.25 4.80 4.23 29.21 0.89 5.44 5.21 14.09 1.16
PSMNet [8] 18.79 13.80 53.22 4.63 23.67 20.61 53.75 5.70 19.75 18.62 42.05 0.94 6.73 5.81 46.24 1.22 6.78 6.40 24.85 1.38
GMStereo [117] 15.63 10.98 46.04 1.87 25.43 22.43 54.70 2.86 6.22 5.58 19.97 0.42 5.68 4.87 38.84 1.10 5.72 5.44 17.33 1.21
ELFNet [59] 24.48 16.94 77.06 8.61 27.08 21.77 85.56 11.01 25.61 24.50 46.06 5.65 10.52 8.67 88.21 2.30 9.61 8.22 85.64 2.16
PCVNet [132] 16.79 13.54 35.66 2.96 12.92 10.19 40.23 2.18 4.24 3.61 14.01 0.41 4.44 3.92 27.70 0.89 5.08 4.88 13.72 1.24
DLNR [140] 9.46 6.20 28.75 1.45 8.44 5.88 32.71 1.24 23.12 22.94 26.93 9.89 9.45 8.83 36.75 1.59 15.74 15.41 34.32 2.83
Selective-RAFT [110] 12.05 9.46 27.42 2.35 15.69 13.86 36.32 5.92 4.36 3.81 10.23 0.34 5.71 5.16 30.54 1.08 6.50 6.22 18.44 1.27
Selective-IGEV [110] 9.98 7.09 27.62 1.60 8.89 6.34 32.88 1.60 6.42 5.71 18.71 1.73 6.22 5.54 34.78 1.09 5.87 5.66 14.99 1.42
IGEV-Stereo [116] 9.91 7.08 26.26 1.84 9.15 6.43 34.88 1.53 4.30 3.86 12.65 0.38 5.65 4.43 33.38 1.03 5.87 5.13 14.31 1.34
NMRF [28] 14.08 10.87 34.62 2.91 23.36 21.69 42.51 8.57 4.34 3.66 17.15 0.42 4.62 4.05 30.65 0.92 5.24 5.07 12.28 1.16
Stereo Anywhere (ours) 6.96 4.75 20.34 0.94 7.97 5.71 29.52 1.08 1.66 1.43 5.29 0.24 3.90 3.52 21.65 0.83 3.93 3.79 11.01 0.97
Table 2. Zero-shot Generalization. Comparison with state-of-the-art deep stereo models. Networks trained on SceneFlow [61].
batches of 2 images. We extract random crops of size
320×640 from images and apply standard color and spatial augmentations [55]. The VFM is used only to source
monocular depth maps, remaining frozen during training.
The number of iterations for GRUs is fixed to 12 during
training and increased to 32 at inference time.
5.2. Evaluation Datasets & Protocol
Datasets. We utilize SceneFlow [61] as our sole training
dataset, comprising about 39k synthetic stereo pairs with
dense ground-truth disparities. For evaluation, we employ
several benchmarks: Middlebury 2014 [86] and its 2021
extension [65] provide high-resolution indoor scenes with
semi-dense labels (15 and 24 stereo pairs), KITTI 2012 [23]
and 2015 [62] feature outdoor driving scenarios (∼200 pairs
each at 1280 × 384 with sparse LiDAR ground truth), and
ETH3D [87] contributes 27 low-resolution indoor/outdoor
scenes. For non-Lambertian surfaces, we primarily use
Booster [127], containing 228 high-resolution (12 Mpx) indoor pairs with its 191-pair online benchmark, and LayeredFlow [115], featuring 400 pairs with transparent objects
and sparse ground truth (∼50 points per pair). Additionally,
we include our newly proposed MonoTrap dataset focusing
on optical illusions. For zero-shot evaluation, we test on
KITTI 2015, Middlebury v3 at half (H) resolution, Middlebury 2021, and ETH3D, while non-Lambertian zero-shot
testing relies on Booster at quarter (Q) resolution and LayeredFlow at eight (E) resolution.
Evaluation Metrics. We evaluate our method using
two standard metrics: the average pixel error (Avg.), which
computes the absolute difference between predicted and
ground truth disparities averaged over all pixels, and the
bad> τ error, which measures the percentage of pixels with
a disparity error greater than τ pixels – for the latter, we
compute it considering all pixels or either non-occluded or
occluded pixels, referred to as All, Noc or Occ respectively.
We evaluate on MonoTrap through standard monocular depth metrics [25] - Absolute relative error (AbsRel),
RMSE, and δ < 1.05 score.
5.3. Ablation Study
We start our analysis by evaluating how individual components of our model contribute to the overall accuracy.
All model variants are trained solely on the synthetic
SceneFlow dataset and tested on Booster and Middlebury
2014, allowing us to examine their effectiveness on nonLambertian surfaces and general scenes.
Table 1 summarizes our findings. In (A), we report the
performance of our baseline model, upon which we build
Stereo Anywhere– i.e., RAFT-stereo [55]. On the one hand,
by adding monocular context from an off-the-shelf monocular depth network to the pre-trained context backbone (B),
we observe improved performance on non-Lambertian surfaces, though at the expense of a general drop in accuracy on Middlebury. On the other hand, by re-training
the context backbone to process depth maps obtained from
the monocular network on SceneFlow (C), we can appreciate a consistent improvement in both datasets. Introducing
the normals correlation volume with subsequent differentiable depth scaling (D) significantly enhances the accuracy
on non-Lambertian surfaces, also showing improvements
on indoor scenes. Finally, cost volume augmentations and
truncation (E) demonstrate positive effects on transparent
surfaces and mirrors present in the Booster dataset by further reducing the bad-2 metric by approximately 1.5% and
Avg. by 0.7 pixels, with minimal influence on Middlebury.
According to these results, from now on, we will adopt
(E) as the default setting for Stereo Anywhere.
6
RGB RAFT-Stereo [55] DLNR [140] NMRF [28] Selective-IGEV [110] Stereo Anywhere
KITTI 15 Middlebury ETH3D
Figure 4. Qualitative Results – Zero-Shot Generalization. Predictions by state-of-the-art models and Stereo Anywhere.
Booster (Q) LayeredFlow (E)
Model Error Rate (%) Avg. Error Rate (%) Avg.
> 2 > 4 > 6 > 8 (px) > 1 > 3 > 5 (px)
RAFT-Stereo [55] 17.84 13.06 10.76 9.24 3.59 89.21 79.02 71.61 19.27
PSMNet [8] 34.47 24.83 20.46 17.77 7.26 91.85 79.84 70.04 21.18
GMStereo [117] 32.44 22.52 17.96 15.02 5.29 92.95 83.68 74.76 20.91
ELFNet [59] 45.52 35.79 30.72 27.33 14.04 93.08 82.24 70.41 20.19
PCVNet [132] 22.63 16.51 13.81 12.08 4.70 88.27 76.65 66.79 18.19
DLNR [140] 18.56 14.55 12.61 11.22 3.97 89.90 79.46 72.72 18.97
Selective-RAFT [110] 20.01 15.08 12.52 10.88 4.12 92.69 86.32 78.82 20.18
Selective-IGEV [110] 18.52 14.24 12.14 10.77 4.38 91.31 81.72 74.74 19.65
IGEV-Stereo [116] 16.90 13.23 11.40 10.20 3.94 87.28 80.07 72.91 19.07
NMRF [28] 27.08 19.06 15.43 13.21 5.02 89.08 79.13 70.51 20.17
Stereo Anywhere (ours) 9.01 5.40 4.12 3.34 1.21 81.83 57.66 45.12 11.20
Booster (Q) Online Benchmark LayeredFlow (E)
DKT-RAFT [136] (*) 10.32 7.13 5.65 4.36 1.70 66.05 46.95 37.77 8.72
Stereo Anywhere (ours) (*) 6.52 2.82 1.77 1.27 0.73 51.24 25.63 15.65 4.84
Table 3. Zero-shot Non-Lambertian Generalization. Comparison with state-of-the-art models. Networks trained on SceneFlow [61].
(*) means fine-tuned on Booster training set.
5.4. Zero-Shot Generalization
We now compare our Stereo Anywhere model against stateof-the-art deep stereo networks, assessing zero-shot generalization capability when transferred from synthetic to real
images. Purposely, we follow a well-established benchmark
in the literature [55, 104], evaluating on real datasets models
pre-trained exclusively on SceneFlow [61].
Table 2 compares Stereo Anywhere with off-the-shelf
stereo networks using authors’ provided weights. Considering All, Noc, and Avg. metrics, we can notice how Stereo
Anywhere achieves consistently better results across most
datasets, achieving almost 3% lower bad-2 All on Middlebury 2014 versus the second-best method DLNR [140], and
breaking the 4% barrier on KITTI’s bad-3 All metric.
The Occ metric further demonstrates how Stereo Anywhere consistently outperforms other stereo models on any
dataset, with substantial margins over the second-best – i.e.,
approximately 6% on Middlebury 2014 and KITTI 2012,
and 3% on ETH3D. This confirms that leveraging priors
from VFMs for monocular depth estimation effectively improve the stereo matching estimation accuracy in challenging conditions where stereo matching is ill-posed, such as
at occluded regions.
Figure 4 shows predictions on KITTI 2015, Middlebury
2014, and ETH3D samples. In particular, the first row
shows an extremely challenging case for SceneFlow-trained
models, where Stereo Anywhere achieves accurate disparity
maps thanks to VFM priors.
5.5. Zero-Shot Non-Lambertian Generalization
We now assess the generalization capabilities of Stereo
Anywhere and existing stereo models when dealing with
non-Lambertian materials, such as transparent surfaces or
mirrors. To this end, we conduct a zero-shot generalization
evaluation experiment on the Booster [74] and LayeredFlow [115] datasets, once again using models pre-trained
on SceneFlow [61] – with weights provided by the authors.
Table 3 shows the outcome of this evaluation. This time,
we can perceive even more clearly how Stereo Anywhere is
the absolute winner, demonstrating unprecedented robustness in the presence of non-Lambertian surfaces despite being trained only on synthetic stereo data, not even featuring
such objects. These results further validate how leveraging
strong priors from existing VFMs for monocular depth esti7
RGB RAFT-Stereo [55] DLNR [140] NMRF [28] Selective-IGEV [110] Stereo Anywhere
Booster LayeredFlow
Figure 5. Qualitative results – Zero-Shot non-Lambertian Generalization. Predictions by state-of-the-art models and Stereo Anywhere.
MonoTrap
Model AbsRel RMSE σ < 1.05
(%)↓ (m)↓ (%)↑
Depth Anything v2 [121] 53.46 0.36 15.21
Depth Anything v2 [121] † 27.92 0.27 19.43
DepthPro [6] 47.77 0.32 21.90
DepthPro [6] † 20.82 0.22 22.88
RAFT-Stereo [55] 5.01 0.09 77.05
Stereo Anywhere 3.50 0.06 80.27
Table 4. MonoTrap Benchmark. Comparison with state-of-theart monocular depth estimation models and RAFT-Stereo. Both
RAFT-Stereo and Stereo Anywhere are trained on SceneFlow
[61]. † refers to robust scaling through RANSAC.
mation can play a game-changing role in stereo matching as
well, especially when lacking training data explicitly targeting critical conditions such as non-Lambertian surfaces. At
the bottom, we report results achieved by fine-tuning Stereo
Anywhere on the Booster training set and evaluating on the
online benchmark. Our model ranks first when evaluated at
quarter resolution.
Figure 5 shows examples from Booster and LayeredFlow, where Stereo Anywhere is the only stereo model correctly perceiving the mirror and transparent railing.
5.6. MonoTrap Benchmark
We conclude our evaluation by running experiments on our
newly collected MonoTrap dataset to prove the robustness
of Stereo Anywhere in the presence of critical conditions
harming the accuracy of monocular depth predictors.
Table 4 collects the results achieved by state-of-the-art
monocular depth estimation models, the baseline stereo
model over which we built our framework (RAFT-Stereo)
and Stereo Anywhere. Regarding the former models, as
they predict affine-invariant depth maps, following the literature [78] we use least square errors to align them to the
ground-truth. As these models are fooled by the visual illusions, this scaling procedure is likely to yield sub-optimal
scale and shift parameters. Therefore, we alternatively align
to ground-truth depth through a more robust RANSAC fitting – denoted with † in the table.
On the one hand, by comparing monocular and stereo
methods, we notice how the failures of the former negatively impact their evaluation metrics. Once again, we reRGB D. Anything v2 [121] Stereo Anywhere
Figure 6. Qualitative results – MonoTrap. Stereo Anywhere is
not fooled by erroneous predictions by its monocular engine [121].
mark that a direct comparison across the two families of
methods is not the main goal of this experiment. On the
other hand, we focus on the comparison between RAFTStereo and Stereo Anywhere, with our model performing
slightly better than its baseline. This fact proves that despite its strong reliance on the priors retrieved from VFMs
for monocular depth estimation, Stereo Anywhere can properly ignore such priors when unreliable.
Figure 6 shows three samples where Depth Anything v2
fails while Stereo Anywhere does not.
6. Conclusion
In this paper, we introduced Stereo Anywhere, a novel
stereo matching framework that leverages monocular depth
VFMs to overcome traditional stereo matching limitations.
Combining stereo geometric constraints with monocular
priors, our approach demonstrates superior zero-shot generalization and robustness to challenging conditions like
textureless regions, occlusions, and non-Lambertian surfaces. Furthermore, through our novel MonoTrap dataset,
we showed that Stereo Anywhere effectively combines the
best of both worlds - maintaining stereo matching’s geometric accuracy where monocular methods fail, while leveraging monocular priors to handle challenging stereo scenarios.
Extensive comparisons against state-of-the-art networks in
zero-shot settings validate these findings.
8
Acknowledgement. This study was carried out within the
MOST – Sustainable Mobility National Research Center and received funding from the European Union Next-GenerationEU –
PIANO NAZIONALE DI RIPRESA E RESILIENZA (PNRR) –
MISSIONE 4 COMPONENTE 2, INVESTIMENTO 1.4 – D.D.
1033 17/06/2022, CN00000023. This manuscript reflects only the
authors’ views and opinions, neither the European Union nor the
European Commission can be considered responsible for them.
This study was funded by the European Union – Next Generation EU within the framework of the National Recovery and Resilience Plan NRRP – Mission 4 “Education and Research” –
Component 2 - Investment 1.1 “National Research Program and
Projects of Significant National Interest Fund (PRIN)” (Call D.D.
MUR n. 104/2022) – PRIN2022 – Project reference: “RiverWatch: a citizen-science approach to river pollution monitoring”
(ID: 2022MMBA8X, CUP: J53D23002260006).
We also acknowledge the CINECA award under the ISCRA
initiative, for the availability of high-performance computing resources and support.
References
[1] Filippo Aleotti, Fabio Tosi, Li Zhang, Matteo Poggi, and
Stefano Mattoccia. Reversing the cycle: self-supervised
deep stereo through enhanced monocular distillation. In
Computer Vision–ECCV 2020: 16th European Conference,
Glasgow, UK, August 23–28, 2020, Proceedings, Part XI
16, pages 614–632. Springer, 2020. 3
[2] Filippo Aleotti, Fabio Tosi, Pierluigi Zama Ramirez, Matteo Poggi, Samuele Salti, Stefano Mattoccia, and Luigi
Di Stefano. Neural disparity refinement for arbitrary resolution stereo. In 2021 International Conference on 3D Vision
(3DV), pages 207–217. IEEE, 2021. 2, 3
[3] Vasileios Arampatzakis, George Pavlidis, Nikolaos Mitianoudis, and Nikos Papamarkos. Monocular depth estimation: A thorough review. IEEE Transactions on Pattern
Analysis and Machine Intelligence, 2023. 1
[4] Antyanta Bangunharcana, Jae Won Cho, Seokju Lee, In So
Kweon, Kyung-Soo Kim, and Soohyun Kim. Correlateand-excite: Real-time stereo matching via guided cost volume excitation. In IEEE/RSJ International Conference on
Intelligent Robots and Systems (IROS), 2021. 2, 4, 16
[5] Luca Bartolomei, Matteo Poggi, Fabio Tosi, Andrea Conti,
and Stefano Mattoccia. Active stereo without pattern projector. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18470–18482, 2023. 2
[6] Aleksei Bochkovskii, Amael Delaunoy, Hugo Germain, ¨
Marcel Santos, Yichao Zhou, Stephan R. Richter, and
Vladlen Koltun. Depth pro: Sharp monocular metric depth
in less than a second. arXiv, 2024. 8, 19, 20, 32
[7] Changjiang Cai, Matteo Poggi, Stefano Mattoccia, and
Philippos Mordohai. Matching-space stereo networks for
cross-domain generalization. In 2020 International Conference on 3D Vision (3DV), pages 364–373, 2020. 2
[8] Jia-Ren Chang and Yong-Sheng Chen. Pyramid stereo
matching network. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition, pages 5410–
5418, 2018. 2, 6, 7
[9] Liyan Chen, Weihan Wang, and Philippos Mordohai.
Learning the distribution of errors in stereo matching for
joint disparity and uncertainty estimation. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17235–17244, 2023. 2
[10] Weifeng Chen, Zhao Fu, Dawei Yang, and Jia Deng.
Single-image depth perception in the wild. In Proceedings
of the 30th International Conference on Neural Information
Processing Systems, page 730–738, Red Hook, NY, USA,
2016. Curran Associates Inc. 3
[11] Xihao Chen, Zhiwei Xiong, Zhen Cheng, Jiayong Peng,
Yueyi Zhang, and Zheng-Jun Zha. Degradation-agnostic
correspondence from resolution-asymmetric stereo. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12962–12971,
2022. 3
[12] Zhi Chen, Xiaoqing Ye, Wei Yang, Zhenbo Xu, Xiao Tan,
Zhikang Zou, Errui Ding, Xinming Zhang, and Liusheng
Huang. Revealing the reciprocal relations between selfsupervised stereo and monocular depth estimation. In Proceedings of the IEEE/CVF International Conference on
Computer Vision (ICCV), pages 15529–15538, 2021. 3
[13] Ziyang Chen, Wei Long, He Yao, Yongjun Zhang, Bingshu Wang, Yongbin Qin, and Jia Wu. Mocha-stereo: Motif
channel attention network for stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, 2024. 2
[14] Junda Cheng, Longliang Liu, Gangwei Xu, Xianqi Wang,
Zhaoxing Zhang, Yong Deng, Jinliang Zang, Yurui Chen,
Zhipeng Cai, and Xin Yang. Monster: Marry monodepth to
stereo unleashes power. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
2025. 3
[15] Kelvin Cheng, Tianfu Wu, and Christopher Healey. Revisiting non-parametric matching cost volumes for robust and
generalizable stereo matching. Advances in Neural Information Processing Systems, 35:16305–16318, 2022. 2
[16] Jaehoon Cho, Dongbo Min, Youngjung Kim, and
Kwanghoon Sohn. Diml/cvl rgb-d dataset: 2m rgb-d images of natural indoor and outdoor scenes. arXiv preprint
arXiv:2110.11590, 2021. 3
[17] WeiQin Chuah, Ruwan Tennakoon, Reza Hoseinnezhad,
Alireza Bab-Hadiashar, and David Suter. Itsa: An
information-theoretic approach to automatic shortcut
avoidance and domain generalization in stereo matching
networks. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pages
13022–13032, 2022. 2
[18] Alex Costanzino, Pierluigi Zama Ramirez, Matteo Poggi,
Fabio Tosi, Stefano Mattoccia, and Luigi Di Stefano.
Learning depth estimation for transparent and mirror surfaces. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9244–9255, 2023. 3
[19] Yiqun Duan, Xianda Guo, and Zheng Zhu. DiffusionDepth:
Diffusion denoising approach for monocular depth estimation. arXiv preprint arXiv:2303.05021, 2023. 3
[20] Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir
Zamir. Omnidata: A scalable pipeline for making multi9
task mid-level vision datasets from 3d scans. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10786–10796, 2021. 3
[21] David Eigen, Christian Puhrsch, and Rob Fergus. Depth
map prediction from a single image using a multi-scale
deep network. In Proceedings of the 27th International
Conference on Neural Information Processing Systems -
Volume 2, page 2366–2374, Cambridge, MA, USA, 2014.
MIT Press. 3
[22] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma,
Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long.
Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image. arXiv preprint
arXiv:2403.12013, 2024. 2, 3
[23] Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we
ready for autonomous driving? the kitti vision benchmark
suite. In 2012 IEEE conference on computer vision and
pattern recognition, pages 3354–3361. IEEE, 2012. 3, 6
[24] Magnus Kaufmann Gjerde, Filip Slezak, Joakim Bruslund ´
Haurum, and Thomas B Moeslund. From nerf to 3dgs: A
leap in stereo dataset quality? In Synthetic Data for Computer Vision Workshop@ CVPR 2024, 2024. 2
[25] Clement Godard, Oisin Mac Aodha, and Gabriel J Bros- ´
tow. Unsupervised monocular depth estimation with leftright consistency. In Proceedings of the IEEE conference on
computer vision and pattern recognition, pages 270–279,
2017. 2, 3, 6
[26] Clement Godard, Oisin Mac Aodha, Michael Firman, and ´
Gabriel J Brostow. Digging into self-supervised monocular
depth estimation. In ICCV, pages 3828–3838, 2019. 3
[27] Rui Gong, Weide Liu, Zaiwang Gu, Xulei Yang, and
Jun Cheng. Learning intra-view and cross-view geometric knowledge for stereo matching. arXiv preprint
arXiv:2402.19270, 2024. 2
[28] Tongfan Guan, Chen Wang, and Yun-Hui Liu. Neural
markov random field for stereo matching, 2024. 2, 6, 7,
8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
[29] Vitor Guizilini, Rui Hou, Jie Li, Rares Ambrus, and
Adrien Gaidon. Semantically-guided representation learning for self-supervised monocular depth. arXiv preprint
arXiv:2002.12319, 2020. 3
[30] Vitor Guizilini, Igor Vasiljevic, Dian Chen, Rares, Ambrus,
,
and Adrien Gaidon. Towards zero-shot scale-aware monocular depth estimation. In ICCV, 2023. 3
[31] Weiyu Guo, Zhaoshuo Li, Yongkui Yang, Zheng Wang,
Russell H Taylor, Mathias Unberath, Alan Yuille, and Yingwei Li. Context-enhanced stereo transformer. In European
Conference on Computer Vision, pages 263–279. Springer,
2022. 2
[32] Xiaoyang Guo, Kai Yang, Wukui Yang, Xiaogang Wang,
and Hongsheng Li. Group-wise correlation stereo network.
In CVPR, 2019. 2
[33] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li,
Kaiqiang Zhou, Hongbo Liu, Bingbing Liu, and Ying-Cong
Chen. Lotus: Diffusion-based visual foundation model for
high-quality dense prediction. In ICLR, 2025. 3, 19, 20
[34] Julia Hornauer and Vasileios Belagiannis. Gradient-based
uncertainty for monocular depth estimation. In European
Conference on Computer Vision, pages 613–630. Springer,
2022. 3
[35] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long,
Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and
Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface
normal estimation. arXiv preprint arXiv:2404.15506, 2024.
3
[36] Wenbo Hu, Xiangjun Gao, Xiaoyu Li, Sijie Zhao, Xiaodong Cun, Yong Zhang, Long Quan, and Ying Shan.
Depthcrafter: Generating consistent long depth sequences
for open-world videos. arXiv preprint arXiv:2409.02095,
2024. 3
[37] Sergio Izquierdo, Mohamed Sayed, Michael Firman,
Guillermo Garcia-Hernando, Daniyar Turmukhambetov,
Javier Civera, Oisin Mac Aodha, Gabriel J. Brostow, and
Jamie Watson. MVSAnywhere: Zero shot multi-view
stereo. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, 2025. 3
[38] Yuanfeng Ji, Zhe Chen, Enze Xie, Lanqing Hong, Xihui Liu, Zhaoqiang Liu, Tong Lu, Zhenguo Li, and Ping
Luo. DDP: Diffusion model for dense visual prediction. In
ICCV, 2023. 3
[39] Hualie Jiang, Zhiqiang Lou, Laiyan Ding, Rui Xu,
Minglang Tan, Wenjie Jiang, and Rui Huang. Defomstereo: Depth foundation model based stereo matching. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025. 3
[40] Junpeng Jing, Jiankun Li, Pengfei Xiong, Jiangyu Liu,
Shuaicheng Liu, Yichen Guo, Xin Deng, Mai Xu, Lai Jiang,
and Leonid Sigal. Uncertainty guided adaptive warping for
robust and efficient stereo matching. In Proceedings of the
IEEE/CVF International Conference on Computer Vision
(ICCV), pages 3318–3327, 2023. 2
[41] Junpeng Jing, Ye Mao, and Krystian Mikolajczyk. Matchstereo-videos: Bidirectional alignment for consistent dynamic stereo matching. In Proceedings of the European
Conference on Computer Vision (ECCV), 2024. 2
[42] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia
Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR), pages 13229–
13239, 2023. 2
[43] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular
depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
2024. 2, 3
[44] Alex Kendall, Hayk Martirosyan, Saumitro Dasgupta, Peter Henry, Ryan Kennedy, Abraham Bachrach, and Adam
Bry. End-to-end learning of geometry and context for deep
stereo regression. In Proceedings of the IEEE international
conference on computer vision, pages 66–75, 2017. 2
[45] Kwonyoung Kim, Jungin Park, Jiyoung Lee, Dongbo Min,
and Kwanghoon Sohn. Pointfix: Learning to fix domain
10
bias for robust online stereo adaptation. In European
Conference on Computer Vision, pages 568–585. Springer,
2022. 3
[46] Marvin Klingner, Jan-Aike Termohlen, Jonas Mikolajczyk, ¨
and Tim Fingscheidt. Self-supervised monocular depth estimation: Solving the dynamic object problem by semantic
guidance. In Computer Vision–ECCV 2020: 16th European
Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XX 16, pages 582–600. Springer, 2020. 3
[47] Hamid Laga, Laurent Valentin Jospin, Farid Boussaid, and
Mohammed Bennamoun. A survey on deep learning techniques for stereo-based depth estimation. IEEE transactions on pattern analysis and machine intelligence, 44(4):
1738–1764, 2020. 1, 2
[48] Iro Laina, Christian Rupprecht, Vasileios Belagiannis, Federico Tombari, and Nassir Navab. Deeper depth prediction
with fully convolutional residual networks. In 2016 Fourth
international conference on 3D vision (3DV), pages 239–
248. IEEE, 2016. 3
[49] Ang Li, Anning Hu, Wei Xi, Wenxian Yu, and Danping Zou. Stereo-lidar depth estimation with deformable
propagation and learned disparity-depth conversion. arXiv
preprint arXiv:2404.07545, 2024. 2
[50] Jiankun Li, Peisen Wang, Pengfei Xiong, Tao Cai, Ziwei
Yan, Lei Yang, Jiangyu Liu, Haoqiang Fan, and Shuaicheng
Liu. Practical stereo matching via cascaded recurrent network with adaptive correlation. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 16263–16272, 2022. 2
[51] Zhengqi Li and Noah Snavely. Megadepth: Learning
single-view depth prediction from internet photos. In
CVPR, pages 2041–2050, 2018. 3
[52] Zhaoshuo Li, Xingtong Liu, Nathan Drenkow, Andy Ding,
Francis X. Creighton, Russell H. Taylor, and Mathias Unberath. Revisiting stereo depth estimation from a sequenceto-sequence perspective with transformers. In Proceedings
of the IEEE/CVF International Conference on Computer
Vision (ICCV), pages 6197–6206, 2021. 2
[53] Zhengfa Liang, Yiliu Feng, Yulan Guo, Hengzhu Liu, Wei
Chen, Linbo Qiao, Li Zhou, and Jianfeng Zhang. Learning
for disparity estimation through feature constancy. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 2811–2820, 2018. 2
[54] Han Ling, Yinghui Sun, Quansen Sun, Ivor Tsang, and
Yuhui Zheng. Self-assessed generation: Trustworthy label generation for optical flow and stereo matching in realworld. arXiv preprint arXiv:2410.10453, 2024. 2
[55] Lahav Lipson, Zachary Teed, and Jia Deng. Raft-stereo:
Multilevel recurrent field transforms for stereo matching.
In International Conference on 3D Vision (3DV), 2021. 1,
2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
29, 30, 31, 32
[56] Biyang Liu, Huimin Yu, and Guodong Qi. Graftnet: Towards domain generalized stereo matching with a broadspectrum and task-oriented feature. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 13012–13021, 2022. 2
[57] Pengpeng Liu, Irwin King, Michael R. Lyu, and Jia Xu.
Flow2stereo: Effective self-supervised learning of optical
flow and stereo matching. In IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2020.
2
[58] Yicun Liu, Jimmy Ren, Jiawei Zhang, Jianbo Liu, and
Mude Lin. Visually imbalanced stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 2029–2038, 2020. 3
[59] Jieming Lou, Weide Liu, Zhuo Chen, Fayao Liu, and
Jun Cheng. Elfnet: Evidential local-global fusion for
stereo matching. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17784–
17793, 2023. 2, 6, 7
[60] Reza Mahjourian, Martin Wicke, and Anelia Angelova. Unsupervised learning of depth and ego-motion from monocular video using 3d geometric constraints. In Proceedings of
the IEEE conference on computer vision and pattern recognition, pages 5667–5675, 2018. 3
[61] Nikolaus Mayer, Eddy Ilg, Philip Hausser, Philipp Fischer,
Daniel Cremers, Alexey Dosovitskiy, and Thomas Brox. A
large dataset to train convolutional networks for disparity,
optical flow, and scene flow estimation. In Proceedings of
the IEEE conference on computer vision and pattern recognition, pages 4040–4048, 2016. 2, 5, 6, 7, 8, 17, 19
[62] Moritz Menze and Andreas Geiger. Object scene flow for
autonomous vehicles. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
3061–3070, 2015. 3, 6
[63] Jaeho Moon, Juan Luis Gonzalez Bello, Byeongjun Kwon,
and Munchurl Kim. From-ground-to-objects: Coarseto-fine self-supervised monocular depth estimation of dynamic objects with ground contact prior. arXiv preprint
arXiv:2312.10118, 2023. 3
[64] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob
Fergus. Indoor segmentation and support inference from
rgbd images. In ECCV, 2012. 3
[65] Guanghan Pan, Tiansheng Sun, Toby Weed, and Daniel
Scharstein. 2021 Mobile stereo datasets with ground truth.
https : / / vision . middlebury . edu / stereo /
data/scenes2021/, 2021. 6
[66] Andrea Pilzer, Yuxin Hou, Niki Loppi, Arno Solin, and
Juho Kannala. Expansion of visual hints for improved
generalization in stereo matching. In Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer
Vision (WACV), pages 5840–5849, 2023. 2
[67] Matteo Poggi and Fabio Tosi. Federated online adaptation
for deep stereo. In CVPR, 2024. 3
[68] Matteo Poggi, Fabio Tosi, and Stefano Mattoccia. Learning monocular depth estimation with unsupervised trinocular assumptions. In 2018 International conference on 3d
vision (3DV), pages 324–333. IEEE, 2018. 3
[69] Matteo Poggi, Davide Pallotti, Fabio Tosi, and Stefano
Mattoccia. Guided stereo matching. In Proceedings of
the IEEE/CVF conference on computer vision and pattern
recognition, pages 979–988, 2019. 2
[70] Matteo Poggi, Filippo Aleotti, Fabio Tosi, and Stefano Mattoccia. On the uncertainty of self-supervised monocular
11
depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
3227–3237, 2020. 3
[71] Matteo Poggi, Alessio Tonioni, Fabio Tosi, Stefano Mattoccia, and Luigi Di Stefano. Continual adaptation for deep
stereo. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 44(9):4713–4729, 2021. 3
[72] Matteo Poggi, Fabio Tosi, Konstantinos Batsos, Philippos
Mordohai, and Stefano Mattoccia. On the synergies between machine learning and binocular stereo for depth estimation from images: a survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):5314–5334,
2021. 1, 2
[73] Michael Ramamonjisoa, Yuming Du, and Vincent Lepetit. Predicting sharp and accurate occlusion boundaries in
monocular depth estimation using displacement fields. In
CVPR, 2020. 3
[74] Pierluigi Zama Ramirez, Fabio Tosi, Matteo Poggi,
Samuele Salti, Stefano Mattoccia, and Luigi Di Stefano.
Open challenges in deep stereo: The booster dataset. In
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21168–21178,
2022. 7
[75] Pierluigi Zama Ramirez, Fabio Tosi, Luigi Di Stefano,
Radu Timofte, Alex Costanzino, Matteo Poggi, Samuele
Salti, Stefano Mattoccia, Jun Shi, Dafeng Zhang, Yong A,
Yixiang Jin, Dingzhe Li, Chao Li, Zhiwen Liu, Qi Zhang,
Yixing Wang, and Shi Yin. Ntire 2023 challenge on hr
depth from images of specular and transparent surfaces.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) Workshops, 2023.
CVPRW. 2
[76] Pierluigi Zama Ramirez, Alex Costanzino, Fabio Tosi, Matteo Poggi, Samuele Salti, Stefano Mattoccia, and Luigi Di
Stefano. Booster: A benchmark for depth from images of
specular and transparent surfaces. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 46(1):85–102,
2024. 2
[77] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi- ´
sion transformers for dense prediction. ICCV, 2021. 3
[78] Rene Ranftl, Katrin Lasinger, David Hafner, Konrad ´
Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot crossdataset transfer. IEEE Transactions on Pattern Analysis and
Machine Intelligence, 44(3), 2022. 3, 8
[79] Anurag Ranjan, Varun Jampani, Lukas Balles, Kihwan
Kim, Deqing Sun, Jonas Wulff, and Michael J Black. Competitive collaboration: Joint unsupervised learning of depth,
camera motion, optical flow and motion segmentation. In
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12240–12249, 2019. 3
[80] Zhibo Rao, Bangshu Xiong, Mingyi He, Yuchao Dai, Renjie He, Zhelun Shen, and Xing Li. Masked representation
learning for domain generalized stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pages 5435–5444, 2023.
2
[81] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ
Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding.
In International Conference on Computer Vision (ICCV)
2021, 2021. 2, 5
[82] Ashutosh Saxena, Min Sun, and Andrew Y. Ng. Make3d:
Depth perception from a single still image. In Proc. AAAI,
2008. 3
[83] Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek
Kar, Mohammad Norouzi, Deqing Sun, and David J. Fleet.
The surprising effectiveness of diffusion models for optical flow and monocular depth estimation. arXiv preprint
arXiv:2306.01923, 2023. 3
[84] Saurabh Saxena, Abhishek Kar, Mohammad Norouzi, and
David J Fleet. Monocular depth estimation using diffusion
models. arXiv preprint arXiv:2302.14816, 2023. 3
[85] Daniel Scharstein and Richard Szeliski. A taxonomy and
evaluation of dense two-frame stereo correspondence algorithms. International journal of computer vision, 47:7–42,
2002. 2
[86] Daniel Scharstein, Heiko Hirschmuller, York Kitajima, ¨
Greg Krathwohl, Nera Nesiˇ c, Xi Wang, and Porter West- ´
ling. High-resolution stereo datasets with subpixel-accurate
ground truth. In Pattern Recognition: 36th German Conference, GCPR 2014, Munster, Germany, September 2-5, ¨
2014, Proceedings 36, pages 31–42. Springer, 2014. 1, 6
[87] Thomas Schops, Johannes L Schonberger, Silvano Galliani,
Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with highresolution images and multi-camera videos. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 3260–3269, 2017. 6
[88] Akihito Seki and Marc Pollefeys. Sgm-nets: Semi-global
matching with neural networks. In Proceedings of the
IEEE conference on computer vision and pattern recognition, pages 231–240, 2017. 2
[89] Jiahao Shao, Yuanbo Yang, Hongyu Zhou, Youmin Zhang,
Yujun Shen, Matteo Poggi, and Yiyi Liao. Learning temporally consistent video depth from video diffusion priors.
arXiv preprint arXiv:2406.01493, 2024. 3
[90] Zhelun Shen, Yuchao Dai, and Zhibo Rao. Cfnet: Cascade
and fused cost volume for robust stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 13906–13915, 2021. 2
[91] Zhelun Shen, Yuchao Dai, Xibin Song, Zhibo Rao, Dingfu
Zhou, and Liangjun Zhang. Pcw-net: Pyramid combination
and warping cost volume for stereo matching. In European
Conference on Computer Vision, pages 280–297. Springer,
2022. 2
[92] Xiao Song, Xu Zhao, Hanwen Hu, and Liangji Fang.
Edgestereo: A context integrated residual pyramid network
for stereo matching. In Computer Vision–ACCV 2018: 14th
Asian Conference on Computer Vision, Perth, Australia,
December 2–6, 2018, Revised Selected Papers, Part V 14,
pages 20–35. Springer, 2019. 2
[93] Xiao Song, Guorun Yang, Xinge Zhu, Hui Zhou, Zhe
Wang, and Jianping Shi. Adastereo: A simple and efficient
12
approach for adaptive stereo matching. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 10328–10337, 2021. 2
[94] Jaime Spencer, Chris Russell, Simon Hadfield, and Richard
Bowden. Kick back & relax: Learning to reconstruct
the world by watching slowtv. In Proceedings of the
IEEE/CVF International Conference on Computer Vision
(ICCV), pages 15768–15779, 2023. 3
[95] Jaime Spencer, Chris Russell, Simon Hadfield, and Richard
Bowden. Kick back & relax++: Scaling beyond groundtruth depth with slowtv & cribstv. arXiv preprint
arXiv:2403.01569, 2024. 3
[96] Aristotle Spyropoulos, Nikos Komodakis, and Philippos
Mordohai. Learning to detect ground control points for improving the accuracy of stereo matching. In Proceedings of
the IEEE conference on computer vision and pattern recognition, pages 1621–1628, 2014. 2
[97] Qing Su and Shihao Ji. Chitransformer: Towards reliable
stereo from cues. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pages 1939–1949, 2022. 2
[98] Yihong Sun and Bharath Hariharan. Dynamo-depth: Fixing unsupervised depth estimation for dynamical scenes. In
Thirty-seventh Conference on Neural Information Processing Systems, 2023. 3
[99] Zachary Teed and Jia Deng. Raft: Recurrent all-pairs
field transforms for optical flow. In Computer Vision–
ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part II 16, pages 402–419.
Springer, 2020. 2
[100] Alessio Tonioni, Matteo Poggi, Stefano Mattoccia, and
Luigi Di Stefano. Unsupervised adaptation for deep stereo.
In Proceedings of the IEEE International Conference on
Computer Vision, pages 1605–1613, 2017. 3
[101] Alessio Tonioni, Fabio Tosi, Matteo Poggi, Stefano Mattoccia, and Luigi Di Stefano. Real-time self-adaptive deep
stereo. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 195–204,
2019. 3
[102] Fabio Tosi, Filippo Aleotti, Pierluigi Zama Ramirez, Matteo Poggi, Samuele Salti, Luigi Di Stefano, and Stefano
Mattoccia. Distilled semantics for comprehensive scene understanding from videos. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition,
pages 4654–4665, 2020. 3
[103] Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger.
Smd-nets: Stereo mixture density networks. In Conference on Computer Vision and Pattern Recognition (CVPR),
2021. 3
[104] Fabio Tosi, Alessio Tonioni, Daniele De Gregorio, and Matteo Poggi. Nerf-supervised deep stereo. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 855–866, 2023. 2, 3, 7
[105] Fabio Tosi, Filippo Aleotti, Pierluigi Zama Ramirez, Matteo Poggi, Samuele Salti, Stefano Mattoccia, and Luigi
Di Stefano. Neural disparity refinement. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.
2, 3
[106] Fabio Tosi, Pierluigi Zama Ramirez, and Matteo Poggi.
Diffusion models for monocular depth estimation: Overcoming challenging conditions. In European Conference
on Computer Vision (ECCV), 2024. 3
[107] Fabio Tosi, Luca Bartolomei, and Matteo Poggi. A survey on deep stereo matching in the twenties. International
Journal of Computer Vision, 2025. 2
[108] Lijun Wang, Jianming Zhang, Yifan Wang, Huchuan Lu,
and Xiang Ruan. CLIFFNet for monocular depth estimation with hierarchical embedding loss. In ECCV. Springer,
2020. 3
[109] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang,
Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking
accurate monocular geometry estimation for open-domain
images with optimal training supervision. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 2025. 3, 19, 20
[110] Xianqi Wang, Gangwei Xu, Hao Jia, and Xin Yang.
Selective-stereo: Adaptive frequency information selection
for stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
2, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
[111] Jamie Watson, Michael Firman, Gabriel J. Brostow, and
Daniyar Turmukhambetov. Self-supervised monocular
depth hints. In ICCV, 2019. 3
[112] Jamie Watson, Oisin Mac Aodha, Daniyar Turmukhambetov, Gabriel J Brostow, and Michael Firman. Learning stereo from single images. In Computer Vision–ECCV
2020: 16th European Conference, Glasgow, UK, August
23–28, 2020, Proceedings, Part I 16, pages 722–740.
Springer, 2020. 3
[113] Philippe Weinzaepfel, Thomas Lucas, Vincent Leroy,
Yohann Cabon, Vaibhav Arora, Romain Bregier, Gabriela ´
Csurka, Leonid Antsfeld, Boris Chidlovskii, and Jer´ ome ˆ
Revaud. CroCo v2: Improved Cross-view Completion Pretraining for Stereo Matching and Optical Flow. In ICCV,
2023. 2
[114] Bowen Wen, Matthew Trepte, Joseph Aribido, Jan Kautz,
Orazio Gallo, and Stan Birchfield. Foundationstereo: Zeroshot stereo matching. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition,
2025. 3
[115] Hongyu Wen, Erich Liang, and Jia Deng. Layeredflow: A
real-world benchmark for non-lambertian multi-layer optical flow. arXiv preprint arXiv:2409.05688, 2024. Accepted
to ECCV 2024. 2, 6, 7
[116] Gangwei Xu, Xianqi Wang, Xiaohuan Ding, and Xin Yang.
Iterative geometry encoding volume for stereo matching.
In Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pages 21919–21928, 2023.
2, 4, 6, 7
[117] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi,
Fisher Yu, Dacheng Tao, and Andreas Geiger. Unifying
flow, stereo and depth estimation. IEEE Transactions on
Pattern Analysis and Machine Intelligence, 2023. 2, 6, 7
[118] Peng Xu, Zhiyu Xiang, Chenyu Qiao, Jingyun Fu, and
Xijun Zhao. Adaptive multi-modal cross-entropy loss for
13
stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.
3
[119] Gengshan Yang, Joshua Manela, Michael Happold, and
Deva Ramanan. Hierarchical deep stereo matching on highresolution images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
5515–5524, 2019. 2
[120] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR,
2024. 2, 3
[121] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024. 1, 2, 3, 5, 8, 19, 20,
32
[122] Wei Yin, Xinlong Wang, Chunhua Shen, Yifan Liu, Zhi
Tian, Songcen Xu, Changming Sun, and Dou Renyin. Diversedepth: Affine-invariant depth prediction using diverse
data. arXiv preprint arXiv:2002.00569, 2020. 3
[123] Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu,
Kaixuan Wang, Xiaozhi Chen, and Chunhua Shen. Metric3D: Towards zero-shot metric 3d prediction from a single
image. In ICCV, 2023. 3
[124] Zhichao Yin and Jianping Shi. Geonet: Unsupervised learning of dense depth, optical flow and camera pose. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 1983–1992, 2018. 3
[125] Zhichao Yin, Trevor Darrell, and Fisher Yu. Hierarchical
discrete distribution decomposition for match density estimation. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition, pages 6044–6053,
2019. 2
[126] Pierluigi Zama Ramirez, Matteo Poggi, Fabio Tosi, Stefano Mattoccia, and Luigi Di Stefano. Geometry meets
semantics for semi-supervised monocular depth estimation. In Computer Vision–ACCV 2018: 14th Asian Conference on Computer Vision, Perth, Australia, December 2–6,
2018, Revised Selected Papers, Part III 14, pages 298–313.
Springer, 2019. 3
[127] Pierluigi Zama Ramirez, Fabio Tosi, Matteo Poggi,
Samuele Salti, Luigi Di Stefano, and Stefano Mattoccia.
Open challenges in deep stereo: the booster dataset. In Proceedings of the IEEE conference on computer vision and
pattern recognition, 2022. CVPR. 1, 2, 6
[128] Pierluigi Zama Ramirez, Alex Costanzino, Fabio Tosi, Matteo Poggi, Luigi Di Stefano, Jean-Baptiste Weibel, Dominik Bauer, Doris Antensteiner, Markus Vincze, Jiaqi
Li, Yachuan Huang, Junrui Zhang, Yiran Wang, Jinghong
Zheng, Liao Shen, Zhiguo Cao, Ziyang Song, Zerong
Wang, Ruijie Zhu, Hao Zhang, Rui Li, Jiang Wu, Xian
Li, Yu Zhu, Jinqiu Sun, Yanning Zhang, Pihai Sun, Yuanqi
Yao, Wenbo Zhao, Kui Jiang, Junjun Jiang, Mykola Lavreniuk, and Jui-Lin Wang. Tricky 2024 challenge on monocular depth from images of specular and transparent surfaces.
In European Conference on Computer Vision Workshops
(ECCVW, 2024. 2
[129] Pierluigi Zama Ramirez, Fabio Tosi, Luigi Di Stefano,
Radu Timofte, Alex Costanzino, Matteo Poggi, et al.
NTIRE 2024 challenge on HR depth from images of
specular and transparent surfaces. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) Workshops, 2024. 2
[130] Jure Zbontar and Yann LeCun. Computing the stereo ˇ
matching cost with a convolutional neural network. In Proceedings of the IEEE conference on computer vision and
pattern recognition, pages 1592–1599, 2015. 2
[131] Jure Zbontar and Yann LeCun. Stereo matching by training ˇ
a convolutional neural network to compare image patches.
Journal of Machine Learning Research, 17(65):1–32, 2016.
2
[132] Jiaxi Zeng, Chengtang Yao, Lidong Yu, Yuwei Wu, and
Yunde Jia. Parameterized cost volume for stereo matching.
In Proceedings of the IEEE/CVF International Conference
on Computer Vision (ICCV), pages 18347–18357, 2023. 2,
6, 7
[133] Jiaxi Zeng, Chengtang Yao, Yuwei Wu, and Yunde Jia.
Temporally consistent stereo matching. In Proceedings
of the European Conference on Computer Vision (ECCV),
2024. 2
[134] Feihu Zhang, Victor Prisacariu, Ruigang Yang, and
Philip HS Torr. Ga-net: Guided aggregation net for end-toend stereo matching. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages
185–194, 2019. 2
[135] Feihu Zhang, Xiaojuan Qi, Ruigang Yang, Victor
Prisacariu, Benjamin Wah, and Philip Torr. Domaininvariant stereo matching networks. In Europe Conference
on Computer Vision (ECCV), 2020. 2
[136] Jiawei Zhang, Jiahe Li, Lei Huang, Xiaohan Yu, Lin Gu,
Jin Zheng, and Xiao Bai. Robust synthetic-to-real transfer
for stereo matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
20247–20257, 2024. 7
[137] Youmin Zhang, Matteo Poggi, and Stefano Mattoccia. Temporalstereo: Efficient spatial-temporal stereo matching network. In IROS, 2023. 2
[138] Yongjian Zhang, Longguang Wang, Kunhong Li, Yun
Wang, and Yulan Guo. Learning representations from foundation models for domain generalized stereo matching. In
Proceedings of the European Conference on Computer Vision (ECCV), 2024. 2
[139] Chaoqiang Zhao, Youmin Zhang, Matteo Poggi, Fabio Tosi,
Xianda Guo, Zheng Zhu, Guan Huang, Yang Tang, and
Stefano Mattoccia. Monovit: Self-supervised monocular
depth estimation with a vision transformer. In 2022 international conference on 3D vision (3DV), pages 668–678.
IEEE, 2022. 3
[140] Haoliang Zhao, Huizhou Zhou, Yongjun Zhang, Jie Chen,
Yitong Yang, and Yong Zhao. High-frequency stereo
matching network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
1327–1336, 2023. 2, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28,
29, 30, 31
14
[141] Tinghui Zhou, Matthew Brown, Noah Snavely, and
David G Lowe. Unsupervised learning of depth and egomotion from video. In Proceedings of the IEEE conference
on computer vision and pattern recognition, pages 1851–
1858, 2017. 3
[142] Yuliang Zou, Zelun Luo, and Jia-Bin Huang. Df-net: Unsupervised joint learning of depth and flow using cross-task
consistency. In Proceedings of the European conference on
computer vision (ECCV), pages 36–53, 2018. 3
15
Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching
Even Where Either Stereo or Mono Fail
Supplementary Material
This document reports additional material concerning CVPR paper “Stereo Anywhere: Robust Zero-Shot Deep Stereo Matching Even
Where Either Stereo or Mono Fail”. Specifically:
• First, we present an extended description of our proposed architecture in Sec. 3, including detailed formulations of the monocular
correlation volume (Sec. 7.1), differentiable monocular scaling (Sec. 7.2), cost volume augmentation (Sec. 7.3), volume truncation
(Sec. 7.4), and training supervision (Sec. 7.5).
• We then report extensive ablation studies in Sec. 8 demonstrating how our stereo matching architecture effectively generalizes across
different state-of-the-art monocular depth networks (Sec. 8.1), showing consistent improvements over baseline stereo methods regardless
of the specific VFM employed. Then, we show qualitatively the impact of the truncated cost volume augmentation on disparity estimation
on non-Lambertian surfaces (Sec. 8.2). Furthermore, we include an analysis of runtime performance and memory consumption (Sec. 8.3)
across different input resolutions and VFMs.
• Finally, we present extensive qualitative results in Sec. 9 across multiple datasets, demonstrating the effectiveness of our method in
dealing with challenging scenarios such as non-Lambertian surfaces, transparent objects and textureless regions.
7. Method Overview: Additional Details
In this section, we enrich the description of Stereo Anywhere architecture.
7.1. Monocular Correlation Volume
Given the monocular depth estimations ML ∈ R
1×H×W and MR ∈ R
1×H×W , we aim to estimate the normal maps ∇L ∈ R
3× H
4 × W
4
and ∇R ∈ R
3× H
4 × W
4 to construct the 3D correlation volume VM ∈ R
H
4 × W
4 × W
4 . We decide to use ∇L and ∇R instead of extracting
additional features from ML and MR because ML and MR already provide high-level information. Furthermore, normal maps can handle
depth inconsistencies between ML and MR that can occur for example when a foreground object is visible only in a single view. We
downsample ML and MR to 1/4 – bilinear interpolation, then we estimate ∇L and ∇R – spatial gradient:
 \nabla = \frac {\nabla ^*}{\lVert \nabla ^* \rVert }, \quad \nabla ^* = \begin {bmatrix} -\frac {\partial \left (\lambda \mathbf {M}_\frac {1}{4}\right )}{\partial x} & -\frac {\partial \left (\lambda \mathbf {M}_\frac {1}{4}\right )}{\partial y} & 1 \\ \end {bmatrix}, \quad \lambda = \frac {1}{10}\cdot \frac {W}{4} \label {eq:normal_estimation} (8)
where λ is a gain factor that is proportional to W, which permits to achieve scale-invariant normal maps.
Given the absence of texture in normal maps, VM will be not ambiguous only in edges. To alleviate this problem, we
segment VM – the relative depth priors from ML and MR: doing so we aim to reduce the ambiguity by forcing the matching
only in similar depth regions (e.g., foreground objects cannot match with background object since the correlation score is
masked to zero). Considering Eq. (3), we calculate masks ML
n
and MR
n
as follows:
 ({\mathcal {M}_L}^n)_{ij} = \begin {cases} 1\ \text {if}\ \frac {n}{N} \leq (\mathbf {M}_L)_{ij} < \frac {n+1}{N}\\ 0\ \text {otherwise} \end {cases} \quad ({\mathcal {M}_R}^n)_{ik} = \begin {cases} 1\ \text {if}\ \frac {n}{N} \leq (\mathbf {M}_R)_{ik} < \frac {n+1}{N}\\ 0\ \text {otherwise} \end {cases} \label {eq:mask} (9)
To further deal with the ambiguity, we improve the 3D Convolutional Regularization model ϕA – an adapted version of
CoEx [4] correlation volume excitation that exploits both views ML, MR:
 ({\mathbf {V'}^s}_M)_{fijk} = \sigma \left (({{\mathbf {f}_L}^s})_{fij}\right ) \odot \sigma \left (({{\mathbf {f}_R}^s})_{fik}\right ) \odot ({\mathbf {V}^s}_M)_{fijk} \label {eq:our_coex} (10)
where V′s
M is the excited volume, σ(·) is the sigmoid function, ⊙ is the element-wise product, VsM ∈ R
F × H
s × W
s × W
s
is an intermediate correlation feature volume at scale s with F features inside module ϕA, fL
s ∈ R
F × H
s × W
s ×1
and fR
s ∈
R
F × H
s ×1× W
s are shallow 2D conv-features extracted from ML and MR downsampled at proper scale.
7.2. Differentiable Monocular Scaling
As detailed in Sec. 3.2, volume VD
M is used also to estimate the coarse disparity maps Dˆ L Dˆ R, while volume VC
M is utilized
to estimate confidence maps Cˆ L Cˆ R. Dˆ L Cˆ L and Dˆ R Cˆ R are used to scale respectively ML and MR. As described in
Eq. (4), we can estimate left disparity from a correlation volume – a softargmax operation on the last W dimension of VD
M
and – the relationship between left disparity and correlation. Here we report an extended version of Eq. (4) with the explicit
formula for softargmax operator:
16
 (\hat {\mathbf {D}}_L)_{ij} = j - \left (\text {softargmax}_L (\mathbf {V}^D_M )\right )_{ij} = j - \sum _{d}^{\frac {W}{4}} d \cdot \frac {e^{(\mathbf {V}^D_M)_{ijd}}}{\sum _{f}^{\frac {W}{4}} e^{(\mathbf {V}^D_M)_{ijf}}} \label {eq:softmax_left2} (11)
At the same time, given the relationship between right disparity and correlation dR = kL − kR we can estimate the right
disparity performing a softargmax on the first W dimension of VD
M:
 (\hat {\mathbf {D}}_R)_{ik} = \left (\text {softargmax}_R(\mathbf {V}^D_M)\right )_{ik} - k = \sum _{d}^{\frac {W}{4}} d \cdot \frac {e^{(\mathbf {V}^D_M)_{idk}}}{\sum _{f}^{\frac {W}{4}} e^{(\mathbf {V}^D_M)_{ifk}}} - k \label {eq:softmax_right} (12)
Disparity maps Dˆ L Dˆ R are used in combination with confidence maps Cˆ L Cˆ R to obtain a robust scaling. We present
an expanded version of the information entropy based confidence estimation (Eq. (5)), with the explicit formula for softmax
operator:
 (\hat {\mathbf {C}}_L)_{ij} = 1 + \frac {\sum _{d}^{\frac {W}{4}} \left (\text {softmax}_L(\mathbf {V}_M^C)\right )_{ijd} \cdot \log _2 \left ( \left (\text {softmax}_L(\mathbf {V}_M^C)\right )_{ijd} \right )}{\log _2(\frac {W}{4})} = 1 + \frac {\sum _{d}^{\frac {W}{4}} \frac {e^{(\mathbf {V}^C_M)_{ijd}}}{\sum _{f}^{\frac {W}{4}} e^{(\mathbf {V}^C_M)_{ijf}}} \cdot \log _2 \left ( \frac {e^{(\mathbf {V}^C_M)_{ijd}}}{\sum _{f}^{\frac {W}{4}} e^{(\mathbf {V}^C_M)_{ijf}}} \right )}{\log _2(\frac {W}{4})} \label {eq:confidence_left2} (13)
In the same way, we estimate right confidence map Cˆ R performing a softmax operation on the first W dimension of VC
M:
 (\hat {\mathbf {C}}_R)_{ik} = 1 + \frac {\sum _{d}^{\frac {W}{4}} \left (\text {softmax}_R(\mathbf {V}_M^C)\right )_{idk} \cdot \log _2 \left ( \left (\text {softmax}_R(\mathbf {V}_M^C)\right )_{idk} \right )}{\log _2(\frac {W}{4})} = 1 + \frac {\sum _{d}^{\frac {W}{4}} \frac {e^{(\mathbf {V}^C_M)_{idk}}}{\sum _{f}^{\frac {W}{4}} e^{(\mathbf {V}^C_M)_{ifk}}} \cdot \log _2 \left ( \frac {e^{(\mathbf {V}^C_M)_{idk}}}{\sum _{f}^{\frac {W}{4}} e^{(\mathbf {V}^C_M)_{ifk}}} \right )}{\log _2(\frac {W}{4})} \label {eq:confidence_right} (14)
To improve the robustness of the scaling, we introduce a softLRC operator to classify occlusions as low-confidence pixels
and consequentially mask out them from Cˆ L and Cˆ R. We define the softLRC operator as follows:
 \text {softLRC}_L(\mathbf {D},\mathbf {D}_R) = \frac {\log \left (1+\exp \left (T_\text {LRC}-\left | \mathbf {D}_L - \mathcal {W}_L(\mathbf {D}_L,\mathbf {D}_R) \right |\right )\right )}{\log (1+\exp (T_\text {LRC}))} \label {eq:softlrc} (15)
where TLRC = 1 is the LRC threshold and WL(DL, DR) is the warping operator that uses the left disparity DL to warp the
right disparity DR into the left view.
Finally, we can estimate the scale sˆ and shift tˆ – a differentiable weighted least-square approach. We report here the
expanded form of Eq. (6):
 \min _{\hat {s}, \hat {t}} \left \lVert \sqrt {\hat {\mathbf {C}}_L}\odot \left [\left (\hat {s}\mathbf {M}_L + \hat {t}\right ) - \hat {\mathbf {D}}_L \right ] \right \rVert _F + \left \lVert \sqrt {\hat {\mathbf {C}}_R}\odot \left [\left (\hat {s}\mathbf {M}_R + \hat {t}\right ) - \hat {\mathbf {D}}_R \right ] \right \rVert _F \label {eq:scale_shift2} (16)
where ∥·∥F denotes the Frobenius norm.
7.3. Cost Volume Augmentations
Volume augmentations are necessary when the training set – e.g., Sceneflow [61] – does not model particularly complex scenarios where a VFM could be useful, for example, when experiencing non-Lambertian surfaces. Without any augmentation
of this kind, the stereo network would simply overlook the additional information from the monocular branch. As detailed
in the main paper, we propose three volume augmentations and a monocular augmentation to overcome this issue. In this
supplementary section, we explain the rationale behind the introduction of each augmentation:
• Volume Rolling: non-Lambertian surfaces such as mirrors and glasses violate the geometry constraints, leading to a high
matching peak in a wrong disparity bin. This augmentation emulates this behavior by shifting some among the matching
peaks to a random position: consequentially, Stereo Anywhere learns to retrieve the correct peak from the other branch.
• Volume Noising and Volume Zeroing: we introduce noise and false peaks into the correlation volume to simulate scenarios
with texture-less regions, repeating patterns, and occlusions.
• Perfect Monocular Estimation: instead of acting inside the correlation volumes, we can substitute the prediction of the
VFM with a perfect monocular map – the ground truth normalized between [0, 1]. This perfect prediction is noise-free and
therefore the monocular branch of Stereo Anywhere will likely gain importance during the training process.
17
7.4. Volume Truncation
The proposed volume truncation strategy further helps Stereo Anywhere to handle mirror surfaces. Here we introduce
additional details about fuzzy operators – useful to make a boolean expression differentiable – and the sigmoid curve used to
truncate the volume VS – the truncate mask (T)ij =
h(Mˆ L)ij > (Dˆ L)ij
∧ (CM)ij i
∨
h
(CM)ij ∧ ¬(Cˆ L)ij i
.
We can replace operators AND (∧), OR (∨), NOT (¬) and GREATER (>) inside T with the fuzzy counterparts
ANDF(A, B) = A · B, ORF(A, B) = A + B − A · B, NOTF(A) = 1 − A and GREATERF(A, B) = σ(A − B), obtaining the fuzzy truncate mask TF:
 \begin {split} (\mathbf {T}_\text {F})_{ij} &= (\mathbf {T}^A_\text {F})_{ij} + (\mathbf {T}^B_\text {F})_{ij} - (\mathbf {T}^A_\text {F})_{ij} \cdot (\mathbf {T}^B_\text {F})_{ij}\\ (\mathbf {T}^A_\text {F})_{ij} &= (\mathbf {C}_M)_{ij} \cdot \sigma \left ( (\hat {M}_L)_{ij} - (\hat {D}_L)_{ij} \right )\\ (\mathbf {T}^B_\text {F})_{ij} &= (\mathbf {C}_M)_{ij} \cdot \left (1-(\hat {\mathbf {C}}_L)_{ij}\right ) \end {split} \label {eq:truncate_mask_fuzzy}
(17)
where TA
F
and TB
F
are respectively the left section and the right section of the ORF of mask TF. Next, we can apply
threshold Tm to achieve the final fuzzy mask T′
F
as follows:
 (\mathbf {T}'_\text {F})_{ij}=\sigma \left ((\mathbf {T}_\text {F})_{ij}-T_m\right ) \label {eq:truncate_mask_fuzzy_thresholded} (18)
.
Finally, we can use the fuzzy truncate mask T′
F
and the scaled monocular map Mˆ L to generate the sigmoid-based truncation volume VT :
 (\mathbf {V}_T)_{ijk} = \left (1-(\mathbf {T}'_\text {F})_{ij}\right ) + (\mathbf {T}'_\text {F})_{ij} \cdot \left [ \sigma \left (j - (\hat {\mathbf {M}}_L)_{ij} - k\right ) \cdot (1-G) + G \right ] \label {eq:truncate_vol} (19)
where G = 0.9 attenuates the impact of the truncation. The correlation volume VS is truncated through an element-wise
product with VT .
7.5. Training Supervision
We supervise the iterative module – the L1 loss with exponentially increasing weights [55]:
 \mathcal {L}_\text {A} = \sum _{l=1}^L{\gamma ^{L-l}\lVert \mathbf {D}^l-\mathbf {D}_\text {Lgt} \rVert _1} \label {eq:loss_raft} (20)
where L is the total number of iterations made by the update operator and DLgt is the ground truth of the left disparity map.
Furthermore, we supervise the outputs Dˆ L, Dˆ R, Mˆ L, Mˆ R, Cˆ L, Cˆ R of the monocular branch – respectively L1 loss and
normal loss for Dˆ L Dˆ R, L1 loss for Mˆ L Mˆ R and Binary Cross Entropy (BCE) loss for Cˆ L Cˆ R:
 \mathcal {L}_\text {B} = \lVert \hat {\mathbf {D}}_L - \mathbf {D}_\text {Lgt} \rVert _1 + \psi \left \lVert \mathbf {1} - \nabla _L\cdot \hat {\nabla }_L \right \rVert _1 \quad \left ( \nabla _L\cdot \hat {\nabla }_L \right )_{ij} = \sum _h (\nabla _L)_{hij} \cdot (\hat {\nabla }_L)_{hij} \label {eq:loss_coarse_disp_left} (21)
 \mathcal {L}_\text {C} = \lVert \hat {\mathbf {D}}_R - \mathbf {D}_\text {Rgt} \rVert _1 + \psi \left \lVert \mathbf {1} - \nabla _R\cdot \hat {\nabla }_R \right \rVert _1 \quad \left ( \nabla _R\cdot \hat {\nabla }_R \right )_{ik} = \sum _h (\nabla _L)_{hik} \cdot (\hat {\nabla }_L)_{hik} \label {eq:loss_coarse_disp_right} (22)
 \mathcal {L}_\text {D} = \lVert \hat {\mathbf {M}}_L - \mathbf {D}_\text {Lgt} \rVert _1 \quad \mathcal {L}_\text {E} = \lVert \hat {\mathbf {M}}_R - \mathbf {D}_\text {Rgt} \rVert _1 \label {eq:loss_scaled_disp} (23)
 \mathcal {L}_\text {F} = \text {BCE}(\hat {\mathbf {C}}_L,\mathbf {C}_\text {Lgt}), \quad (\mathbf {C}_\text {Lgt})_{ij} = \frac {\log \left (1+\exp \left (T_\text {LRC}-\left | (\hat {\mathbf {D}}_L)_{ij} - (\mathbf {D}_\text {Lgt})_{ij} \right |\right )\right )}{\log (1+\exp (T_\text {LRC}))} \label {eq:loss_coarse_conf_left} (24)
 \mathcal {L}_\text {G} = \text {BCE}(\hat {\mathbf {C}}_R,\mathbf {C}_\text {Rgt}), \quad (\mathbf {C}_\text {Rgt})_{ik} = \frac {\log \left (1+\exp \left (T_\text {LRC}-\left | (\hat {\mathbf {D}}_R)_{ik} - (\mathbf {D}_\text {Rgt})_{ik} \right |\right )\right )}{\log (1+\exp (T_\text {LRC}))} \label {eq:loss_coarse_conf_right} (25)
where ψ = 10 is the normal loss weight, DRgt is the ground truth of the right disparity map, ∇ˆ L ∇ˆ R are the normal maps
estimated respectively from Dˆ L Dˆ R, ∇L · ∇ˆ L and ∇R · ∇ˆ R are the dot product between normal maps, and CLgt CRgt are the
confidence ground truth. The final supervision loss L is the sum of all previous partial losses:
 \mathcal {L} = \mathcal {L}_A + \mathcal {L}_B + \mathcal {L}_C + \mathcal {L}_D + \mathcal {L}_E + \mathcal {L}_F + \mathcal {L}_G \label {eq:total_loss} (26)
18
8. Additional Ablation Studies
In this section, we report additional studies concerning the performance of Stereo Anywhere.
8.1. Generalization to Different VFMs
In the main paper, we assumed Depth Anything v2 [121] as the VFM fueling Stereo Anywhere, since it is the latest stateof-the-art model being published at the time of this submission. However, any VFM for monocular depth estimation would
be suitable for this purpose, either current or future ones. To confirm this argument, we conducted some experiments by
replacing Depth Anything v2 with other VFMs that appeared on arXiv in the last months, yet that are not been officially
published. Among them, we select DepthPro [6], MoGe [109] and Lotus [33].
Table 5 shows the results achieved by Stereo Anywhere variants – different VFMs on Booster and LayeredFlow. We can
appreciate how the different flavors of Stereo Anywhere always outperform the baseline stereo model [55]. In general, Depth
Anything v2 remains the best choice to deal with non-Lambertian surfaces, with DepthPro allowing for small improvements
on some metrics over the LayeredFlow dataset.
Booster (Q) LayeredFlow (E)
Model Error Rate (%) Avg. Error Rate (%) Avg.
> 2 > 4 > 6 > 8 (px) > 1 > 3 > 5 (px)
Baseline [55] 17.84 13.06 10.76 9.24 3.59 89.21 79.02 71.61 19.27
Stereo Anywhere – DAv2 [121] 9.01 5.40 4.12 3.34 1.21 81.83 57.66 45.12 11.20
Stereo Anywhere – DepthPro [6] 10.53 7.02 5.79 5.13 2.40 78.76 61.11 51.04 14.43
Stereo Anywhere – MoGe [109] 9.47 5.77 4.49 3.84 1.44 84.27 68.67 58.89 16.22
Stereo Anywhere – Lotus [33] 12.44 8.71 7.58 6.98 3.21 86.04 62.75 49.47 13.98
Table 5. Non-Lambertian Generalization of Stereo Anywhere w.r.t VFMs. We measure the impact of different monocular depth
estimation networks. Networks trained on SceneFlow [61].
Table 6 shows the results achieved by the different VFMs on the zero-shot generalization benchmark. Also in this case,
we can appreciate how any Stereo Anywhere variant yields comparable accuracy, with some VFMs like Moge yielding
some improvements over Depth Anything v2 on ETH3D, KITTI 2012 and 2015 at the expense of lowering the accuracy on
Middlebury 2014 and 2021. Interestingly, we can observe an important drop in accuracy by using DepthPro on Middlebury
2021, due to several failures by the model itself on the scenes of this dataset.
Middlebury 2014 (H) Middlebury 2021 ETH3D KITTI 2012 KITTI 2015
Model bad > 2 Avg. bad > 2 Avg. bad > 1 Avg. bad > 3 Avg. bad > 3 Avg.
All Noc Occ (px) All Noc Occ (px) All Noc Occ (px) All Noc Occ (px) All Noc Occ (px)
Baseline [55] 11.15 8.06 29.06 1.55 12.05 9.38 37.89 1.81 2.59 2.24 8.78 0.25 4.80 4.23 29.21 0.89 5.44 5.21 14.09 1.16
Stereo Anywhere – DAv2 [121] 6.96 4.75 20.34 0.94 7.97 5.71 29.52 1.08 1.66 1.43 5.29 0.24 3.90 3.52 21.65 0.83 3.93 3.79 11.01 0.97
Stereo Anywhere – DepthPro [6] 6.58 4.32 20.05 0.99 15.13 12.52 41.16 8.97 2.74 2.54 6.09 0.44 3.13 2.25 18.25 0.75 3.79 3.10 10.53 0.95
Stereo Anywhere – MoGe [109] 7.79 5.23 22.86 1.21 9.86 7.30 33.48 1.28 1.28 1.09 3.78 0.21 2.85 2.00 17.40 0.73 3.22 2.57 8.97 0.89
Stereo Anywhere – Lotus [33] 7.35 4.96 21.71 1.07 9.62 7.01 34.92 1.29 2.68 2.44 6.04 0.31 4.54 3.58 22.71 0.92 3.88 3.21 10.36 0.95
Table 6. Generalization of Stereo Anywhere w.r.t VFMs. We measure the impact of different monocular depth estimation networks.
Networks trained on SceneFlow [61].
Finally, Figure 7 shows qualitative results obtained by the different variants of Stereo Anywhere, highlighting only minor
differences among the different predictions.
RGB RAFT-Stereo [55]
Stereo Anywhere Stereo Anywhere Stereo Anywhere Stereo Anywhere
– DAv2 [121] – DepthPro [6] – MoGe [109] – Lotus [33]
Figure 7. Qualitative Results – Booster and LayeredFlow. Predictions by RAFT-Stereo and Stereo Anywhere – different VFMs.
19
8.2. Impact of Cost Volume Truncation
Cost volume truncation is a specific augmentation we apply to improve the results in the presence of mirrors. Figure 8 shows
a qualitative example of predictions by Stereo Anywhere (using Depth Anything v2) obtained by either not applying or by
applying such augmentation. While Stereo Anywhere alone cannot entirely restore the surface of the mirror starting from the
priors provided by the VFM, applying cost volume truncation allows for predicting a much smoother and consistent surface.
RGB Stereo Anywhere Stereo Anywhere
w/o volume truncation w/ volume truncation
Figure 8. Qualitative Results – Volume Truncation. Predictions by Stereo Anywhere.
8.3. Runtime & Memory Consumption Analysis
Table 7 reports the processing time (in seconds) and memory consumption (in GB) required by Stereo Anywhere during
inference, comparing it with the baseline stereo backbone, RAFT-Stereo. We measure the runtime on a single A100 GPU,
repeating the experiment with three different input resolutions, specifically 256×256, 512×512, and 1024×1024, as well as
by deploying the different VFMs studied before to fuel Stereo Anywhere – specifically, for each variant we report standalone
runtime and memory usage by the VFM and the stereo backbone separately, as well as their sum.
Concerning runtime, Depth Anything v2 is the fastest among the VFMs, taking about 30ms to process a single image
at any resolution, with Moge requiring more than 10× the time for a single inference when processing 1Mpx images. The
stereo backbone requires about 50% additional time compared to the baseline, RAFT-Stereo [55], because of the additional
branch deployed to process the depth maps by the VFM.
For what concerns memory consumption, once again Depth Anything v2 is the most efficient among the VFMs, requiring
as few as 2GB, with Moge sharing similar requirements. Our stereo backbone introduces additional memory consumption
because of the second branch processing monocular cues: this overhead is negligible with 256 images, raising to about 2×
the memory required by RAFT-Stereo alone when dealing with 1Mpx images.
Image Size Stereo Model Name VFM Name Processing Time (s) Memory Consumption (GB)
(H × W) VFM Stereo Total VFM Stereo Total
256 × 256 Stereo Anywhere (ours)
DAv2 [121] 0.03 0.15 0.18 0.57 0.18 0.76
DepthPro [6] 0.21 0.15 0.36 1.92 0.18 2.09
MoGe [109] 0.38 0.15 0.52 0.38 0.19 0.57
Lotus [33] 0.13 0.15 0.29 0.22 0.18 0.41
256 × 256 RAFT-Stereo [55] - - 0.10 0.10 - 0.17 0.17
512 × 512 Stereo Anywhere (ours)
DAv2 [121] 0.03 0.21 0.24 0.57 0.77 1.34
DepthPro [6] 0.20 0.21 0.41 1.84 0.77 2.60
MoGe [109] 0.38 0.21 0.59 0.38 0.78 1.17
Lotus [33] 0.16 0.22 0.38 0.85 0.77 1.62
512 × 512 RAFT-Stereo [55] - - 0.14 0.14 - 0.66 0.66
1024 × 1024 Stereo Anywhere (ours)
DAv2 [121] 0.03 0.61 0.63 0.58 5.73 6.31
DepthPro [6] 0.21 0.61 0.82 1.85 5.73 7.59
MoGe [109] 0.38 0.60 0.98 0.42 5.77 6.19
Lotus [33] 0.49 0.61 1.10 3.40 5.73 9.13
1024 × 1024 RAFT-Stereo [55] - - 0.36 0.36 - 2.63 2.63
Table 7. Runtime & Memory Consumption Analysis.
20
9. Qualitative Results
We conclude with additional qualitative results by Stereo Anywhere on the different datasets involved in our experiments.
Figure 9 shows two examples from the KITTI 2012 dataset (respectively, stereo pairs 000040 and 000068). We can notice
how any existing stereo model is unable to properly perceive the presence of transparent surfaces, as in correspondence of the
windows on buildings and cars. On the contrary Stereo Anywhere, driven by the priors injected through the VFM, properly
predicts the disparity corresponding to the transparent surfaces.
RGB RAFT-Stereo [55]
DLNR [140] NMRF [28]
Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55]
DLNR [140] NMRF [28]
Selective-IGEV [110] Stereo Anywhere (ours)
Figure 9. Qualitative Results – KITTI 2012 (part 1). Predictions by state-of-the-art models and Stereo Anywhere.
21
Figure 10 shows two further examples from KITTI 2012 (respectively, stereo pairs 000073 and 000127). In this case,
we can appreciate the much higher level of detail in the disparity maps predicted by Stereo Anywhere, with extremely thin
structures in fences and gates being preserved.
RGB RAFT-Stereo [55]
DLNR [140] NMRF [28]
Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55]
DLNR [140] NMRF [28]
Selective-IGEV [110] Stereo Anywhere (ours)
Figure 10. Qualitative Results – KITTI 2012 (part 2). Predictions by state-of-the-art models and Stereo Anywhere.
22
Figure 11 reports two stereo pairs from KITTI 2015 (respectively, 000024 and 000049). These examples confirm the
ability to recover both thin structures and transparent surfaces already appreciated in KITTI 2012.
RGB RAFT-Stereo [55]
DLNR [140] NMRF [28]
Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55]
DLNR [140] NMRF [28]
Selective-IGEV [110] Stereo Anywhere (ours)
Figure 11. Qualitative Results – KITTI 2015 (part 1). Predictions by state-of-the-art models and Stereo Anywhere.
23
Figure 12 reports two additional samples from KITTI 2015 (respectively, 000093 and 000144). These latter present both
underexposed and transparent regions, respectively on the billboard and the tram in the two images. While existing stereo
networks struggle at dealing with both, Stereo Anywhere exposes unprecedented robustness.
RGB RAFT-Stereo [55]
DLNR [140] NMRF [28]
Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55]
DLNR [140] NMRF [28]
Selective-IGEV [110] Stereo Anywhere (ours)
Figure 12. Qualitative Results – KITTI 2015 (part 2). Predictions by state-of-the-art models and Stereo Anywhere.
24
Figure 13 reports two image pairs from Middlebury 2014 (respectively, Adirondack and Vintage). On the former, Stereo
Anywhere preserves the very thin holes on the back of the chair, while on the latter it can properly estimate the disparity for
the displays, where existing methods are fooled and predict holes.
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
Figure 13. Qualitative Results – Middlebury 2014. Predictions by state-of-the-art models and Stereo Anywhere.
25
Figure 14 and 15 shows the results on two samples from Middlebury 2021, peculiar for their aspect ratio (respectively,
ladder1 and ladder2). Although existing models perform quite well on both, they fail to preserve the skittles on the top of
the scene, whereas Stereo Anywhere properly predicts their structure.
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
Figure 14. Qualitative Results – Middlebury 2021 (part 1). Predictions by state-of-the-art models and Stereo Anywhere.
26
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
Figure 15. Qualitative Results – Middlebury 2021 (part 2). Predictions by state-of-the-art models and Stereo Anywhere.
27
Figure 16 collects three outdoor images from ETH3D (respectively, Playground1, Playground2 and Playground3). Once
again, Stereo Anywhere proves its supremacy at predicting fine details such as branches and poles, while resulting more
robust to challenging illumination conditions such as the sun flare in Playground2.
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55] DLNR [140]
Figure 16. Qualitative Results – ETH3D. Predictions by state-of-the-art models and Stereo Anywhere.
28
Figure 17 and 18 report four examples from the Booster dataset, confirming how Stereo Anywhere can exploit the strong
priors provided by the VFM to properly perceive the glass surface on the window in the former image, as well as challenging,
untextured black surfaces of the computer, the TV and the displays appearing in the remaining samples.
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
Figure 17. Qualitative Results – Booster (part 1). Predictions by state-of-the-art models and Stereo Anywhere.
29
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
Figure 18. Qualitative Results – Booster (part 2). Predictions by state-of-the-art models and Stereo Anywhere.
30
Figure 19 showcases three images from the LayeredFlow dataset, highlighting once again the inability of the state-of-theart networks to model even small, transparent surfaces as those in the doors from the first and second samples, conversely
to Stereo Anywhere which can properly identify their presence. Finally, the third sample further highlights the high level of
detail in Stereo Anywhere predictions once again.
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
RGB RAFT-Stereo [55] DLNR [140]
NMRF [28] Selective-IGEV [110] Stereo Anywhere (ours)
Figure 19. Qualitative Results – LayeredFlow. Predictions by state-of-the-art models and Stereo Anywhere.
31
To conclude, Figure 20 collects three scenes from our novel MonoTrap dataset. In this case, we report predictions by both
state-of-the-art monocular and stereo models, as well as by Stereo Anywhere. The perspective illusions fooling monocular
methods, unsurprisingly, do not affect stereo networks, which however are inaccurate near the left border of the image (first
sample) or in the absence of texture (second sample). Stereo Anywhere effectively combines the strength of both worlds,
while being not affected by any of their weaknesses.
RGB Depth Anything v2 [121] DepthPro [6]
RAFT-Stereo [55] Selective-IGEV [110] Stereo Anywhere (ours)
RGB Depth Anything v2 [121] DepthPro [6]
RAFT-Stereo [55] Selective-IGEV [110] Stereo Anywhere (ours)
RGB Depth Anything v2 [121] DepthPro [6]
RAFT-Stereo [55] Selective-IGEV [110] Stereo Anywhere (ours)
Figure 20. Qualitative Results – MonoTrap. Predictions by state-of-the-art models and Stereo Anywhere.
32
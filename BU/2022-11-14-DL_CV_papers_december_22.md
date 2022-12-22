---
title: 'Random DL/CV papers'
date: 2022-11-14
permalink: /posts/DL_CV_papers_december_22
tags:
  - arxiv
  - Deep Learning
  - Computer Vision
  - Research
  - Papers
---

Random recent interesting DL/CV papers: two papers on self supervised learning for vision, training captioning models without images, training various vision and language models with inage, HiLo attention, Revenge of the DeiT and a leverging CLIP for model explainability.  



As the title suggests, in this blog post I'll survey random interesting papers I came across. There's no real common theme connecting the papers, aside from me finding them cool and being published this recently 🤷‍♂️. I do try to give a relatively detailed overview so the reader can get the gist of work, but I definitely may skip some details and possibly make errors, so feel free to comment, especially if you're one of the authors. 

CAN - A simple, efficient and scalable contrastive masked autoencoder for learning visual representations 
======
[Arxiv](https://arxiv.org/abs/2210.16870), Keywords: semi-supervised learning, contrastive learning, Venue: under review

In my opinion, a very neat paper that combines several ideas from self supervised learning (SSL), namely contrastive loss (most notably, SimCLR [1]) reconstruction of masked patches (most notably "Masked Auto-encoders Are Scalable Vision Learners" [2]) and denoising autoencoder. Their pipeline is summarized in the figure below and works as follows: given an input image, generate 2 different views by applying augmentations, mask 50% of the patches, add Gaussian noise to the unmasked patches and feed the resulting noisy masked image to a ViT encoder. Now, we take the encoding of the unmasked patches, perform mean pooling, pass to a light MLP and apply contrastive loss (hence, the "contrastive" in the title). The encoded "patch image" is then passed to a ViT decoder to perform reconstruction of the masked patches and denoising of the unmasked noisy patches, which gives us the both reconstruction loss and the denoising loss. 


| ![CAN](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/The_CAN_framework.png) | 
|:--:| 
| <b>The CAN framework:</b>Two views of an image are generated, 50% of patches randomly masked in each, and noise is added to patches. An encoder is trained to solve three tasks: 1) <b>Reconstruction:</b> encoded patches are passed to a decoder that reconstructs missing patches, 2) <b>Denoise:</b> reconstructs the noise added to unmasked patches, and 3) <b>Contrast:</b> pooled patches are passed to a contrastive loss, using in-batch samples as negatives |


Motivated by diffusion transformers[3], the method provides the decoder with information about the noise level. Now, as the noise is modelled a simple zero mean Gaussian with standard deviation $\sigma$, the noise level information can be simply encoded by taking a sinusoidal embedding of $\sigma$, passing it to a light MLP to produce a (learned) embedding for $\sigma$ which is added to the noised patches before feeding those to the decoder. Below is a figure describing this process:


| ![CAN denoising](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/denoising3.png) | 
|:--:| 
| <b>Denoising:</b> Both the encoded patches and the noise level $\sigma$ are passed to the decoder by passing $\sigma$ through an MLP, and adding the result to each embedded token. |

The authors provide an ablation of this component which demonstrate that simply adding noise as an augmentation also improves the performance of the system even without the denoising loss. However, adding the denoising loss without incorporating the noise level information provides worse results while incorporating it outperforms noise augmentation, demonstrating the necessity of this component (see table 1 and ablation discussion in section 3.4).


The results demonstrate improved or on-par performance with recent SSL methods as measured on ImageNet 1K when finetuning or when using linear probing, both with and without pre-training on JFT-300 [4] or on Imagenet [5]:

<img src='https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/CAN_table2.png'> <br/>
<b>JFT-300M pre-training:</b> Comparison to the state of the art on ImageNet linear probe. CAN outperforms all methods except DnC, which uses a complicated multi-stage training process. Computation is measured as ImageNet-equivalent epochs. †Our implementation.

<img src='https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/CAN_table3.png'> <br/>
<b>Pre-training on ImageNet-1K</b> 

The results also show that the method scales well to JFT-300, hence the "scalable" part of the title. The method is also faster than methods which use the full image views, as it only "uses" 50% of the tokens in both views of the image (as opposed to SimCLR for example which augments the entire image) and does not use multiple views per image (such as DINO [6] or SwAV [7] which uses multi-crop), hence the "efficient" in the title. The paper is overall simple and elegant, does not use momentum encoder, hence the "simple" in the title. I should point out that it does not beat all other methods on all datasets, but the overall trade-off between results and simplicity is very good in my opinion. This is the main selling point of the paper: combining different semi-supervised techniques in a way which complement each other to obtain a unifed *simple* and *efficient* system. Keep in mind that the method can also be extended by adding a multiple views, momentum encoder or a tokenizer and a masking objective (as in BeiT[8]) to further improve the results, of course with the cost of complexity and slower running times. 


MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis
====== 



Text-Only Training for Image Captioning using Noise-Injected CLIP
======
[Arxiv](https://arxiv.org/abs/2211.00575), [Code](https://github.com/DavidHuji/CapDec), Keywords: CLIP, Image Captioning, NLP, Venue: EMNLP 2022


Cool paper, simple and elegant. Training image captioning models commonly requires supervision in the form of image and text pairs. Given a text-only dataset (so no images and certainly no pairs), can we leverage CLIP's [9] strong image and text embedding capabilities to train a text-only image captioning model? turns out we can.

As a reminder, given an image *I* with a corresponding text *T*, CLIP embeds *I* and *T* to a shared space where their embeddings are close. If we had image-text pairs, we could learn a decoder that given the CLIP image embedding as a starting point, reconstruct the text. However, in the above settings we don't have access to images at training time, so the authors propose to use the text embedding as a proxy for the image embedding instead. Specifically, given a dataset of sentences, we extract CLIP text embeddings from each sentence and learn a decoder which reconstruct the text from the text embedding and in inference time simply apply the decoder on the input image embedding instead.

This simple baseline performs poorly, as there is a gap between the image and the text embedding - the decoder is trained with the text embeddings, but in test time is applied to the image embeddings, which are close to the text embeddings, but not in the same position (for a given image-text pair).

Let us assume that for each image-text pair, the image embedding (which is given at test time) resides in a small epsilon neighbourhood around the text embedding. If we can learn a decoder that given a text embedding, decodes all vectors in its epsilon neighbourhood to the corresponding text, it would correctly decode the image embedding as well as it resides in its epsilon neighborhood. This is done in by adding a zero-mean Gaussian noise vector with STD epsilon to the text embedding during training. The value of epsilon is selected by taking the mean infinity norm between image embeddings and text embeddings of 15 images from MS-COCO [10]. The authors also provide an ablation study measuring the effect of epsilon on the performance (spoiler: pretty robust to the value of epsilon, as long as it's not too deviated from the "correct" value, like an order of magnitude). Below is a figure providing an overview of the method, which the authors dubbed <b> CapDec </b>:

| ![CapDec overview](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/capdec_method.png) | 
|:--:| 
| <b>Overview of CapDec captioning approach:</b> . <b>(a)</b> An illustration of the CLIP joint embedding space. Embedded text is relatively close to its corresponding visual embedding, but with a certain gap. <b>(b)</b> CapDec trains a model that decodes the CLIP embedding of text *T* back to text *T*, after noise injection. The encoders remain frozen. <b>(c)</b> At inference, CapDec simply decodes the embedding of an image using the trained decoder. |

The method is tested on MS-COCO[10] and Flickr 30k[11] image captioning benchmarks and demonstrates a large improvement over other unsupervised or weakly supervised method. The method of course performance worse than state of the art supervised methods, but as an anecdote I checked, and it actually slightly outperforms "Show, Attend and Tell"[12] which was one of the seminal papers on image captioning.

| ![CapDec results for image captioning](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/capdec_table1.png) | 
|:--:| 
| <b>Results for image captioning:</b> . <b>(A)</b> We use captions from the COCO and Flickr30k to train CapDec and evaluate on the datasets the captions were taken from. We report results for fully supervised methods that train on captioned images, and on methods that use no training text (ZeroCap), or just training text and no images (CapDec and MAGIC). <b>(B)</b> Similar setting to (A), but in cross-domain setup where training text is taken from one dataset, and evaluation is done on the second dataset. |


The authors also show strong performance on style-guided image captioning, a task where the method requires to generate captions in a certain text style in which labeled image-text data can be limited. Those results are summarised below along with several examples:

| ![CapDec Style-Guided captioning results on FlickrStyle10K](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/capdec_table2.png) | 
|:--:| 
| <b>Style-Guided captioning results on FlickrStyle10K:</b> |


| ![CapDec Style-Guided captioning results on FlickrStyle10K](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/capdec_examples.png) | 
|:--:| 
| <b>Example for styled captions of CapDec on FlickrStyle10K</b> |

I Can't Believe There's No Images! Learning Visual Tasks Using only Language Data
======



Fast Vision Transformers with HiLo Attention
======
[arxiv](https://arxiv.org/abs/2205.13213),  [code](https://github.com/ziplab/LITv2) , keywords: Vision Transformers, Venue: Neurips 2022 spotlight paper

The paper proposes a novel efficient ViT architecture with throughput in mind to mitigate ViT's high computational complexity which stems from the quadratic memory and time complexity of the attention mechanism. 

First, the paper argue (and in my opinion rightfully so) that although many improved and more efficient ViT architectures have been proposed, in practice they do not offer high processing speed. This claim might seem contradictory, but in fact previous works usually consider metrics such as number of FLOPS, memory usage and asymptotic computational complexity (which are important by themselves), but those metrics do no capture the actual running time or throughput nor those works directly measure those. Moreover, specific architectures with small number of FLOPS and memory requirements or lower asymptotic complexity as might actually run slowly when implemented on GPU due to specific operations which not hardware-friendly or cannot be parallelized. To this end, the paper directly benchmarks FLOPS and memory consumption as well as throughput (on GPU) and proposes a ViT architecture that performs favourably in those metrics while achieving high accuracy when used as backbone in classification and various down-stream vision tasks. 

The proposed ViT architecture is based on changing the attention mechanism by seperating the self-attention heads into two groups. One group (1-$\alpha$) of the heads performs self-attention in local windows on the original high resolution feature map (denoted <i>Hi-Fi attention</i>), thus capturing fine details in small local windowns (characterised by high frequencies) while the second group perfoms regular global self attention but on a downscaled (max-pooled) version of the feature map (denoted <i>Lo-Fi attention</i>) to captured global structures (characterised by low frequencies). The features maps from the two groups are concatenated and passed to the following HiLo attention block. 

| ![HiLo attention framework](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/Hilo_figure1.png) | 
|:--:| 
| <b>Framework of HiLo attention:</b>  $N_h$ refers to the total number of self-attention heads at this layer. $\alhpa$ denotes the split ratio for high/low frequency heads. |

The authors provide an ablation study measuring the effect of different choices of $\alpha$ (see figure below). As $\alpha$ increases, the fraction of heads allocated to the second group performing global attention on the downscaled feature map increases, bringing more "attention" (apologies for the "notation overloading") to global structures. This also reduces FLOPS and improves the running time as Lo-Fi attention has lower computational complexity than Hi-Fi attention. The authors find that the best performance is obtained when $\alpha=0.9$, meaning 90% of the heads perform global attention on the downscaled features maps and only 10% of the heads attend to local fine details. Interestingly, setting $\alpha=1.0$, meaning essentially removing the Hi-Fi attention and replacing the method with regular attention on downscaled feature maps performs competitively on ImageNet1K, but the authors report it provides worse results on dense prediction tasks such as semantic segmentaion (however, the authors do not provide an ablation using semgenatic segmenation, as far as I can tell).

| ![HiLo attention - effect of alpha](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/Hilo_figure4.png) | 
|:--:| 
| Effect of $\alpha$ based on LITv2-S |

The authors compare the proposed architecture with recent ViT and Convnet architectures on as backbone for Image Classification on ImageNet-1K, Object Detection and Instance Segmentation on COCO and Semantic Segmentation on ADE20K, demonstrating on-par (or better) accuracy against state-of-the-art methods while providing high throughput and a small memory footprint.


| ![HiLo attention - results on Imagenet 1K](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/Hilo_table1.png) | 
|:--:| 
| <b> Image classification results on ImageNet-1K:</b> By default, the FLOPs, throughput and memory consumption are measured based on the resolution 224 × 224 with a batch size of 64. Throughput is tested on one NVIDIA RTX 3090 GPU and averaged over 30 runs. ResNet results are take from "ResNet Stikes Back" [20], “↑ 384” means a model is finetuned at the resolution 384 × 384. “OOM” means “out-of-memory”.|



| ![HiLo attention - Object detection and instance segmentation results](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/Hilo_table2.png) | 
|:--:| 
| <b> Object detection and instance segmentation performance:</b> performance is measured on the COCO val2017 split using the RetinaNet and Mask R-CNN framework. $AP^b$ and $AP^m$ denote the bounding box AP and mask AP, respectively. * indicates the model adopts a local window size of 4 in HiLo.|


| ![HiLo attention - Semantic segmenation results](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/Hilo_table3.png) | 
|:--:| 
| Semantic segmentation performance of different backbones on the ADE20K validation set. FLOPs is evaluated based on the image resolution of 512 × 512.|


The authors further compare their proposed architecture against a wider array of more recent and stronger ViT architectures implemented across various different GPU models. The HiLo transformer achieves higher throughput (i.e.: faster running times) than all other methods on across all GPU models while still acheiving almost the highest top-1 accuracy on ImageNet-1K.

TODO: add table 6


Overall, the paper presents a simple and effieicnt ViT attention machanism, with an emaphsis on having a "GPU friendly" implementation that achieves high throughput. There was a similar paper presented in ECCV with the same motivation, that considers effieicnt ViT architecures for edge devices: EdgeViTs: Competing Light-Weight CNNs on Mobile Devices with Vision Transformers ([arxiv](https://arxiv.org/abs/2205.03436)). 

TODO: add table 10



DeiT III: Revenge of the ViT
======
[Arxiv](https://arxiv.org/abs/2204.07118) [Code](https://github.com/facebookresearch/deit/blob/main/README_revenge.md) Keywords: Vision Transformers, Training recipe, Venue: ECCV 2022

As hinted by the title, the paper is a follow up work to DeiT (Training data-efficient image transformers & distillation through attention [13]) and co-authored by several of DeiT's authors. The goal of the paper is to provide an improved training recipe for “vanilla” Vision Transformers (ViT) [14] in order to achieve a stronger baseline for vanilla ViT, without any architectural changes. I find this extremely interesting as there is a large body of works which offer various architectural changes (some motivated by ConvNets) to vanilla ViT (e.g.: PVT[15], Swin[16], CvT[17], Focal Transformer[18], Coatnet[19]), and here the authors steer away from making any changes to the architecture and focus instead only on the training recipe. This work is also similar to a previous paper by the several of same authors, “ResNet strikes back: An improved training procedure in timm” [20] which offers an improved training receipt for ResNets to achieve a stronger baseline for simple "vanilla" ResNets. Fun fact, there is no DeiT2 ! 

DeiT III is sort of a response to several lines of work: improved ViT architectures such as Swin [16], improved ConvNet architecture such as ConvNext [21] and self-supervised training methods for ViT such as BEiT [8]. The paper suggest several training strategies that improve ViT performance such that training scales to larger model size without saturating, training time is reduced and the final models reach better or on par performance with Swin[16], ConveNext[21] and other recent architecture as well using BeiT[8] like training when benchmarked on ImageNet 1K, ImageNet 21K and downstream tasks. 

The training strategy is composed of following techniques:
* Stochastic Depth [22] which randomly drops layers in the network during training. 
* LayerScale[23] which normalizes each channel of the matrix produced by Multi-Head Self Attention (MSHA) and Feed Forward Network (FFN) blocks using a different learned constant. 
* Replacing Cross Entropy (CE) with Binary Cross Entropy similarly to [20] which provides an improvement in some of the experiments. 
Using the LAMB [24] optimizer.
* 3-Augment: a simple augmentation method composed of either grayscaling, solarization or Gaussian blur (with equal probability) followed by color jittering and horizontal flip. 
* Simple Random Crop: which resizes the input image such that the smallest side matches the training resolution and randomly samples square crops in that resolution. 

Below is a table summarizing the training recipe, including all hyperparameters and compares it to previous methods:


| ![DeiT3 training recipe](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/deit3_table1.png) | 
|:--:| 
|Comparison of Deit3 training recipes including all hyperparameters|

The paper presents several experiments demonstrating the effectiveness of the improved training recipe. First, they show a significant improvement gap compared to vanilla ViT and DeiT training recipes, measured on ImageNet 1k and ImageNet 21k: 

| ![DeiT3 performance compared to previous training recipes](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/deit3_figure1.png) | 
|:--:| 
|Comparison of training recipes for (left) vanilla vision transformers trained on ImageNet-1k and evaluated at resolution 224×224, and (right) pre-trained on ImageNet-21k at 224×224 and finetuned on ImageNet-1k at resolution 224×224 or 384×384.|


In addition, the paper demonstrates on-par performance with recent architectures, such as ConvNext and Swin, measured on ImageNet 1k and ImageNet 21k, see tables below:

| ![DeiT3 classification with ImageNet 1K training](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/deit3_table7.png)| 
|:--:| 
|<b> Classification with Imagenet-1k training: </b> The authors compare architectures with comparable FLOPs and number of parameters. All models are trained on ImageNet1k only without distillation nor self-supervised pre-training. We report Top-1 accuracy on the validation set of ImageNet1k and ImageNetV2 with different measure of complexity: throughput, FLOPs, number of parameters and peak memory usage. The throughput and peak memory are measured on a single V100-32GB GPU with batch size fixed to 256 and mixed precision. The reslts ResNet and RegNet from the "Resnet strikes back" paper [20].  Note that different models may have received a different optimization effort, so this is not a complete "apples to apples" comparison. ↑R indicates that the model is fine-tuned at the resolution R and -R indicates that the model is trained at resolution R. | 



| ![DeiT3 classification with ImageNet 21K training](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/deit3_table8.png)| 
|:--:| 
|<b> Classification with Imagenet-21k training: </b> The authors compare architectures with comparable FLOPs and number of parameters. All models are trained on ImageNet1k only without distillation nor selfsupervised pre-training. We report Top-1 accuracy on the validation set of ImageNet1k and ImageNetV2 with different measure of complexity: throughput, FLOPs, number of parameters and peak memory usage. The throughput and peak memory are measured on a single V100-32GB GPU with batch size fixed to 256 and mixed precision. For Swin-L the authors decrease the batch size to 128 in order to avoid out of memory error and re-estimate the memory consumption. ↑R indicates that the model is fine-tuned at the resolution R. | 


As written above, the paper is also a response to self-supervised training methods, thus the authors also compared their improved *supervised* training recipe to self-supervised alternatives, specifically MAE [25] and BeiT[8]: 

| ![DeiT3 training recipe compared to MAE and BeiT](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/deit3_table9.png)| 
|:--:| 
|<b> Comparison of self-supervised pre-training with DeiT3 approach. </b> Please note that DeiT3 training is full supervised where as MAE and BeiT are self-supervised pre-training methods, so this is not a direct comparison of fully supervised training recipes, but more of trainining "stratagies". All models are evaluated at resolution 224 × 224. Results are reported on ImageNet Val, real and v2 in order to evaluate overfitting. The superscripts (21k) and (1k) indicate finetuning with labels on ImageNet-1k and Imagenet-21k, respectively. |

The paper also demonstrates improved performance in transfer learning on semantic segmentation, measured on ADE20k [26] dataset:

| ![ADE20k semantic segmentation](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/deit3_table11.png)| 
|:--:| 
|<b> ADE20k semantic segmentation. </b> All models are pre-trained on ImageNet-1k except models with † symbol that are pre-trained on ImageNet-21k. The authors report the pre-training resolution used on ImageNet-1k and ImageNet-21k. |

All in all, at first sight DeiT 3 might seem like a “bag of tricks” sort of paper and one might argue that it does not hold enough technical novelty to be presented at a top-tier conference such as ECCV. In my opinion, this is hardly the case. While the novelty is limited (and the authors do not argue otherwise in the text), saying “hey, you can get really good results with vanilla ViT just by improving the training procedure with no architectural changes (or bells and whisles)” is a strong contribution in my opinion. 



CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks
====== 



  
References
======

[1] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020.

[2] He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[3] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in Neural Information Processing Systems 33 (2020).

[4] Sun, Chen, et al. "Revisiting unreasonable effectiveness of data in deep learning era." Proceedings of the IEEE international conference on computer vision. 2017.

[5] Deng, Jia, et al. "Imagenet: A large-scale hierarchical image database." 2009 IEEE conference on computer vision and pattern recognition, https://www.image-net.org/ 

[6] Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[7] Caron, Mathilde, et al. "Unsupervised learning of visual features by contrasting cluster assignments." Advances in Neural Information Processing Systems 33 (2020).
  
[8] Bao, Hangbo, et al. "BEiT: BERT Pre-Training of Image Transformers." International Conference on Learning Representations. 2021.

[9] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International Conference on Machine Learning. PMLR, 2021.

[10] Lin, Tsung-Yi, et al. "Microsoft coco: Common objects in context." European conference on computer vision. Springer, Cham, 2014.

[11] Young, Peter, et al. "From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions." Transactions of the Association for Computational Linguistics 2 (2014). 

[12] Xu, Kelvin, et al. "Show, attend and tell: Neural image caption generation with visual attention." International conference on machine learning. PMLR, 2015.

[13] Touvron, Hugo, et al. "Training data-efficient image transformers & distillation through attention." International Conference on Machine Learning. PMLR, 2021.
  
[14] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020).
  
[15] Wang, Wenhai, et al. "Pyramid vision transformer: A versatile backbone for dense prediction without convolutions." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
  
[16] Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
  
[17] Wu, Haiping, et al. "Cvt: Introducing convolutions to vision transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
  
[18] Yang, Jianwei, et al. "Focal self-attention for local-global interactions in vision transformers." arXiv preprint arXiv:2107.00641 (2021). 
  
[19] Dai, Zihang, et al. "Coatnet: Marrying convolution and attention for all data sizes." Advances in Neural Information Processing Systems 34 (2021).
  
[20]  Wightman, Ross, Hugo Touvron, and Hervé Jégou. "Resnet strikes back: An improved training procedure in timm." arXiv preprint arXiv:2110.00476 (2021).
  
[21] Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
  
[22] Huang, Gao, et al. "Deep networks with stochastic depth." European conference on computer vision. Springer, Cham, 2016.
  
[23] Touvron, Hugo, et al. "Going deeper with image transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
  
[24] You, Yang, et al. "Large batch optimization for deep learning: Training bert in 76 minutes." arXiv preprint arXiv:1904.00962 (2019)
  
[25] He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
  
[26] Zhou, Bolei, et al. "Scene parsing through ade20k dataset." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.


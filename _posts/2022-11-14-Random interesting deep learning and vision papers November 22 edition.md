---
title: 'Random papers November 2022 edition'
date: 2022-11-14
permalink: /posts/random_papers_nov_22
tags:
  - arxiv
  - Deep Learning
  - Computer Vision
  - Research
  - Papers
---

Random (interesting) papers: Contrastive masked autoencoder, training captioning models without images, improved training scheme for ViT and more. 


As the title suggests, here I'll survey random interesting papers I came across. There's no real common theme connecting the papers, aside from me finding them really cool and having published this month 🤷‍♂️. I do try to give a relatively detailed overview so the reader can get the gist of work, but I definitely may skip things and possibly made errors, so feel free to comment, especially if you're by any chance one of the authors. 

CAN - A simple, efficient and scalable contrastive masked autoencoder for learning visual representations 
======
[Arxiv](https://arxiv.org/abs/2210.16870) [code] subjects: semi-supervised learning, contrastive learning

In my opinion, a pretty neat paper that combines several ideas from self supervised learning (SSL), namely contrastive loss (most notably, SimCLR [1]) reconstruction of masked patches (most notably "Masked Autoencoders Are Scalable Vision Learners" [2]) and denoising autoencoder. Their pipeline is summarized in the fiture below and works as follows: take an input image, generate 2 different views by applying augmentations, mask 50% of the patches, add Gaussian noise to the unmasked patches and feed the resulting noisy masked image to a ViT encoder.  Now, we take the encoding of the unmasked patches, perform mean pooling, pass to a light MLP and perform apply contrastive loss (hence, the "contrastive" in the title). The encoded "patch image" is passed to a ViT decoder to perform reconstruction of the masked patches and denoising of the unmasked noisy patches, which gives us the both reconstruction loss and the denoising loss. 

Motivated by diffusion transformers[] , the method provides the decoder with information about the noise level. Now, as the noise is modelled a simple zero mean Gaussian with standard deviation sigma, the noise level information is simply encoded by taking a sinusoidal embedding of sigma, passing it to a light MLP to produce a (learned) embedding for sigma which is added to the noised patches before feeding them to the decoder. The authors provide an abalation of this compoenent which demonstrate that simply adding noise as an augmentation improves the performance of the system even without the denoising loss and that adding the denoising loss improves it even further, but only when the decoder is "informed" by the nosie level. Adding the denosing loss without incorporating the noise level information  provides worse results. The authors do not motivate it in the paper, TODO: add explanation. 



The results demonstrate improved or on-par performance with recent SSL methods as measured on ImageNet 1K when finetuning or when using linear probing, both with pre-training on JFT-300 [] and pre-training on Imagenet []. The result shows that it scales well to JFT-300, hence the "scalable" part of the title. The method is also faster than methods which use the full image views, as it only "uses" 50% of the tokens in both views of the image (as opposed to SimCLR for example which augmentes the entire image) and does not use multiple views per image (such as DINO [3] or SwAV [4] which uses multi-crop to reduce the additional memory use), hence the "efficient" in the title. The paper is overall simple and elegant, does not use momentum encoder, hence the "simple" in the title. I should point out that it does not beat all other methods on all datasets, but the overall trade-off between results and simplicity is very good in my opinion. This isthe main selling point of the paper: combining different semi-supervised techniqes in a way which complement each other to obtain a unifed *simple* and *efficient* systmem. Keep in mind thatthe mtehod can also be exteneded by adding a multiple views, momentum encoder or a tokenizer and using clasisifcation instead of a regression loss, similarity yo Beit






Text-Only Training for Image Captioning using Noise-Injected CLIP
======

Cool paper, simple and elegant. Training image captioning models commonly requires supervision in the form of image and text pairs. Given a text-only dataset (so no images and certainly no pairs), can we leverage CLIP's [5] strong image and text embedding capabilities to train a text-only image captioning model? turns out we can.

As a reminder, given an image I with a corresponding text T, CLIP embeds I and T to a shared space where their embeddings are close. If we would have image-text pairs, we could learn a decoder that given the CLIP image embedding as a starting point, reconstruct the text. However, in the above settings we don't have access to images at training time, so the authors propose to use the text embedding as a proxy instead. Specifically, given a dataset of sentences, we extract CLIP text embeddings from each sentence and learn a decoder which reconstruct the text from the embedding and in inference time simply apply the decoder on the input image embedding instead.

This simple baseline performs poorly, as there is a gap between the image and the text embedding - the decoder is trained with the text embeddings, but in test time is applied to the image embeddings, which are close to the text embeddings, but not in the same position (for a given image-text pair).

Let us assume that for each image-text pair, the image embedding (which is given at test time) resides in a small epsilon neighbourhood around the text embedding. If we would learn a decoder that given a text embedding, decodes all vectors in its epsilon neighbourhood to the corresponding text, it would correctly decode the image embedding to the text as well. This is done in by adding a zero-mean Gaussian noise vector with STD epsilon to the text embedding during training. The value of epsilon is selected by taking the mean infinity norm between image embeddings and text embeddings of 15 images from MS-COCO [6]. The authors also provide an ablation study measuring the effect of epsilon on the performance (spoiler: pretty robust to the value of epsilon, as long as it's not too deviated from the "correct" value, like an order of magnitude).

The method is tested on MS-COCO and Flickr 30k[7]  image captioning benchmarks and demonstrates a large improvement over other unsupervised or weakly supervised method.

The method of course performance worse than state of the art supervised methods, but as an anecdote, I checked and it actually slightly outperforms "Show, Attend and Tell"[8] which was one of the seminal papers on image captioning.

The authors also show strong performance on style-guided image captioning, a task where the method requires to generate captions in a certain text style in which labeled image-text data can be limited.

DeiT III: Revenge of the ViT
======
I’ll start with DeiT III: Revenge of the ViT. The goal of the paper is to provide an improved training recipe for “vanilla” ViT in order to achieve a stronger baseline for vanilla ViT, without any architectural changes. I find this extremely interesting as there is a large body of works which offer “engineered like” architectural changes to vanilla ViT (perhaps most notable is the Swin transformer), and here the authors steer away from making any changes to the architecture and focus instead only on the training recipe. This work is also similar to a previous paper also co-authored by Hugo Touvron, “ResNet strikes back: An improved training procedure in timm” [9] which offers an improved training receipt for ResNets to achieve a stronger baseline and of course is a follow to the first DeiT [1] paper. Fun fact, there is no DeiT2 ! 

DeiT III is sort of a response to several lines of work: improved ViT architectures such as Swin [9], improved ConvNet architecture such as ConvNext [10] and self-supervised training methods for ViT such as BEiT [11]. The paper suggest several training strategies that improve ViT performance such that training scales to larger model size without saturating, training time is reduced and the final models reach better or on par performance with Swin[9], ConveNext[10] and other recent architecture as well using BeiT[11] like training when benchmarked on ImageNet 1K, ImageNet 21K and downstream tasks. The training strategy is composed of following techniques:
Stochastic Depth [12] which randomly drops layers in the network during training. 
LayerScale[4] which normalizes each channel of the matrix produced by Multi-Head Self Attention (MSHA) and Feed Forward Network (FFN) blocks using a different learned constant. 
Replacing Cross Entropy (CE) with Binary Cross Entropy similarly to [9] which provides an improvement in some of the experiments. 
Using the LAMB [13] optimizer.
3-Augment: a simple augmentation method composed of either grayscaling, solarization or Gaussian blur (with equal probability) followed by color jittering and horizontal flip. 
Simple Random Crop: which resizes the input image such that the smallest side matches the training resolution and randomly samples square crops in that resolution. 

Below is a table summarizing the training recipe, including all hyperparameters and compares it to previous methods:

The paper presents several experiments demonstrating the effectiveness of the improved training recipe. First, they show a significant improvement gap compared to vanilla ViT and DeiT training recipes, measured on ImageNet 1k and ImageNet 21k:


In addition, the paper demonstrates on-par performance with recent architectures, such as ConvNext and Swin, measured on ImageNet 1k and ImageNet 21k, see tables below:


The paper also demonstrates improved performance in transfer learning on semantic segmentation, measured on ADE20k [14] dataset:


All in all, at first sight DeiT 3 might seem like a “bag of tricks” sort of paper and one might argue that it does not hold enough technical novelty to be presented at a top-tier conference such as ECCV. In my opinion, this is hardly the case. While the novelty is limited (and the authors do not argue otherwise in the text), saying “hey, you can get really good results with vanilla ViT just by playing with the training a bit, no need for any bells and whistles” is a strong contribution. 


Fast Vision Transformers with HiLo Attention
======


The paper addresses efficient Vision Transformers (ViTs) design. The paper argues that while previous works on designing efficient ViTs have considered the theoretical asymptotic computational complexity and computational complexity measured in floating point operations (FLOPS) and memory, those metrics do not capture the actual running time and throughput. Specifically, the paper argues that previous methods might require low number of FLOPs (or lower asymptotic complexity), but in practise their implementation is not hardware friendly thus slow when running on GPU. The paper proposes to benchmark FLOPS, memory consumption and actual running time (on GPU) and further proposes a ViT design that performs favourably in those metrics while providing high accuracy when used as a backbone in various vision tasks, namely: image classification, object detection, instance segmentation and semantic segmentation.

The proposed ViT architecture is based on separating the MultiHead Self Attention (MSA) heads into 2 groups - one group performs local window self attention to capture local fine grained details characterised by high frequencies and the second group performs global self attention on a downscaled (in practice - average pooling in each high res window) version of the feature map to capture global structures characterised by low frequencies. The total number of MSA heads are divided between the groups such that 1-alpha of the heads belong to the first group (local windowed self attention on the full resolution feature map) and alpha of the heads belong to the second group (global attention on the downscaled feature map). Their method is thus dubbed HiLo to denote the different attention branches working on High and Low frequencies. Regarding the value of alpha - the authors provide an experiment to measuring the effect of different choices of alpha and when measuring on the various benchmarks alpha is set to 0.9, so in practice 10% and 90% of the MSA heads belong to the high and low frequencies branch, respectively. Note that in the low frequency brach, keys and values are computed on the downscaled feature map, but the queries still come from the high frequency branch. Also, to further speed up the method, the authors replace explicit positional encoding by adding a layer of 3x3 depth-wise convolution in each Feed Forward block.

Finally, to demonstrate the effectiveness of their approach, the authors compare their methods to other ViT architecture in classification on ImageNet 1K as well as when using their architecture as a backbone (weights are initialised from the ImageNet trained model) in object detection and instance segmentation (measured on COCO) and semantic segmentation (measured on ADE20K). The experiments demonstrate relatively high speed, low number of FLOPS and high accuracy of the proposed method compared to other ViT architectures and efficient attention mechanisms.


References
======

[1] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020.

[2] He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[3] Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[4] Caron, Mathilde, et al. "Unsupervised learning of visual features by contrasting cluster assignments." Advances in Neural Information Processing Systems 33 (2020): 9912-9924.

[5] 


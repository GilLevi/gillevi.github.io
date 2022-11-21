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

Random (interesting) papers: Contrastive masked autoencoder, training captioning models without images, improved training scheme for ViT, new ViT architecture, unified Vision and Language learning, and more  


As the title suggests, in this blog post I'll survey random interesting papers I came across. There's no real common theme connecting the papers, aside from me finding them cool and being published this recently 🤷‍♂️. I do try to give a relatively detailed overview so the reader can get the gist of work, but I definitely may skip some details and possibly make errors, so feel free to comment, especially if you're one of the authors. 

CAN - A simple, efficient and scalable contrastive masked autoencoder for learning visual representations 
======
[Arxiv](https://arxiv.org/abs/2210.16870), Keywords: semi-supervised learning, contrastive learning

In my opinion, a very neat paper that combines several ideas from self supervised learning (SSL), namely contrastive loss (most notably, SimCLR [1]) reconstruction of masked patches (most notably "Masked Auto-encoders Are Scalable Vision Learners" [2]) and denoising autoencoder. Their pipeline is summarized in the figure below and works as follows: given an input image, generate 2 different views by applying augmentations, mask 50% of the patches, add Gaussian noise to the unmasked patches and feed the resulting noisy masked image to a ViT encoder. Now, we take the encoding of the unmasked patches, perform mean pooling, pass to a light MLP and apply contrastive loss (hence, the "contrastive" in the title). The encoded "patch image" is then passed to a ViT decoder to perform reconstruction of the masked patches and denoising of the unmasked noisy patches, which gives us the both reconstruction loss and the denoising loss. 


| ![CAN](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/The_CAN_framework.png) | 
|:--:| 
| <b>The CAN framework:</b>Two views of an image are generated, 50% of patches randomly masked in each, and noise is added to patches. An encoder is trained to solve three tasks: 1) <b>Reconstruction:</b> encoded patches are passed to a decoder that reconstructs missing patches, 2) <b>Denoise:</b> reconstructs the noise added to unmasked patches, and 3) <b>Contrast:</b> pooled patches are passed to a contrastive loss, using in-batch samples as negatives |


Motivated by diffusion transformers[3], the method provides the decoder with information about the noise level. Now, as the noise is modelled a simple zero mean Gaussian with standard deviation $\sigma$, the noise level information can be simply encoded by taking a sinusoidal embedding of $\sigma$, passing it to a light MLP to produce a (learned) embedding for $\sigma$ which is added to the noised patches before feeding those to the decoder. Below is a figure describing this process:

<img src='https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/denoising2_jpeg.jpg'> <br/>
<b>Denoising:</b> Both the encoded patches and the noise level $\sigma$ are passed to the decoder by passing $\sigma$ through an MLP, and adding the result to each embedded token.

The authors provide an ablation of this component which demonstrate that simply adding noise as an augmentation also improves the performance of the system even without the denoising loss. However, adding the denoising loss without incorporating the noise level information provides worse results while incorporating it outperforms noise augmentation, demonstrating the necessity of this component (see table 1 and ablation discussion in section 3.4).


The results demonstrate improved or on-par performance with recent SSL methods as measured on ImageNet 1K when finetuning or when using linear probing, both with and without pre-training on JFT-300 [4] or on Imagenet [5]:

<img src='https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/CAN_table2.png' width='400' height='400'/> <br/>
<b>JFT-300M pre-training:</b> Comparison to the state of the art on ImageNet linear probe. CAN outperforms all methods except DnC, which uses a complicated multi-stage training process. Computation is measured as ImageNet-equivalent epochs. †Our implementation.

<img src='https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/CAN_table3.png' width='400' height='400'/> <br/>
<b>Pre-training on ImageNet-1K</b> 

The results also show that the method scales well to JFT-300, hence the "scalable" part of the title. The method is also faster than methods which use the full image views, as it only "uses" 50% of the tokens in both views of the image (as opposed to SimCLR for example which augments the entire image) and does not use multiple views per image (such as DINO [6] or SwAV [7] which uses multi-crop), hence the "efficient" in the title. The paper is overall simple and elegant, does not use momentum encoder, hence the "simple" in the title. I should point out that it does not beat all other methods on all datasets, but the overall trade-off between results and simplicity is very good in my opinion. This is the main selling point of the paper: combining different semi-supervised techniques in a way which complement each other to obtain a unifed *simple* and *efficient* system. Keep in mind that the method can also be extended by adding a multiple views, momentum encoder or a tokenizer and a masking objective (as in BeiT[8]) to further improve the results, of course with the cost of complexity and slower running times. 


Text-Only Training for Image Captioning using Noise-Injected CLIP
======
[Arxiv](https://arxiv.org/abs/2211.00575), [Code](https://github.com/DavidHuji/CapDec), Keywords: CLIP, Image Captioning, NLP

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
| <b>Results for image captioning:</b> . <b>(A)</b> We use captions from the COCO and Flickr30k to train CapDec and evaluate on the datasets the captions were taken from. We report results for fully supervised methods that train on captioned images, and on methods that use no training text (ZeroCap), or just training text and no images (CapDec and MAGIC). <b>(B)</b> Similar setting to (A), but in cross-domain setup where training text is taken from one dataset, and evaluation is done on the second dataset.


The authors also show strong performance on style-guided image captioning, a task where the method requires to generate captions in a certain text style in which labeled image-text data can be limited. Those results are summarised below along with several examples:

| ![CapDec Style-Guided captioning results on FlickrStyle10K](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/capdec_table2.png) | 
|:--:| 
| <b>Style-Guided captioning results on FlickrStyle10K:</b> 


| ![CapDec Style-Guided captioning results on FlickrStyle10K](https://github.com/GilLevi/gillevi.github.io/blob/master/_posts/random_papers_nov22/capdec_examples.png) | 
|:--:| 
| <b>Example for styled captions of CapDec on FlickrStyle10K</b> 



DeiT III: Revenge of the ViT
======
As hinted by the title, the paper is a follow up work to DeiT (Training data-efficient image transformers & distillation through attention[13]) and co-authored by several of DeiT's authors. The goal of the paper is to provide an improved training recipe for “vanilla” Vision Transformers (ViT) [14] in order to achieve a stronger baseline for vanilla ViT, without any architectural changes. I find this extremely interesting as there is a large body of works which offer various architectural changes (some motivated by ConvNets) to vanilla ViT (e.g.: PVT[15], Swin[16], CvT[17], Focal Transformer[18], Coatnet[19]), and here the authors steer away from making any changes to the architecture and focus instead only on the training recipe. This work is also similar to a previous paper by the several of same authors, “ResNet strikes back: An improved training procedure in timm” [20] which offers an improved training receipt for ResNets to achieve a stronger baseline for simple "vanilla" ResNets. Fun fact, there is no DeiT2 ! 

DeiT III is sort of a response to several lines of work: improved ViT architectures such as Swin [16], improved ConvNet architecture such as ConvNext [21] and self-supervised training methods for ViT such as BEiT [8]. The paper suggest several training strategies that improve ViT performance such that training scales to larger model size without saturating, training time is reduced and the final models reach better or on par performance with Swin[16], ConveNext[21] and other recent architecture as well using BeiT[8] like training when benchmarked on ImageNet 1K, ImageNet 21K and downstream tasks. 

The training strategy is composed of following techniques:
* Stochastic Depth [12] which randomly drops layers in the network during training. 
* LayerScale[4] which normalizes each channel of the matrix produced by Multi-Head Self Attention (MSHA) and Feed Forward Network (FFN) blocks using a different learned constant. 
* Replacing Cross Entropy (CE) with Binary Cross Entropy similarly to [9] which provides an improvement in some of the experiments. 
Using the LAMB [13] optimizer.
* 3-Augment: a simple augmentation method composed of either grayscaling, solarization or Gaussian blur (with equal probability) followed by color jittering and horizontal flip. 
* Simple Random Crop: which resizes the input image such that the smallest side matches the training resolution and randomly samples square crops in that resolution. 

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


UniCL: Unified Contrastive Learning in Image-Text-Label Space
======
A very cute paper that tries to unified regular supervised learning and text and image contrastive learning. There isn't a lot of novelty in the paper, but the approach and some of the experiments are very interesting, specifically where it uses regular labels as text and check if that improves results is very interesting. 

Basically they say that every image-text pair has a unique label and every image-label pair can be views as several image-text pairs with texts such as "an image of a <label>", "a photo of a <label>", "a <label>".

Table 3 is very interesting where they show that by considering labels as text in the loss function you can get an improvement and if you encoder the label text using a more sophisticated text embedding (e.g. transformer instead of just a simple fully connecter layer label embedding) you can get an additional improvement. 

Explanations on some parts in the paper:

Equation 2 - the i'th text can correspond to more than one image (since we have stuff like "a photo of a chihuahua" so all images of chihuahua correspond to it), so denote by P(i) the indices in the batch that corresponds to the label of the i'th text and k goes over all P(i).

Same for equation 3 - the j'th image can correspond to more than one text (since we have stuff like a "photo of a chihuahua" and "an image of a chihuahua" both corresponding to the same image), so denote by by p(j) the indices in the batch that corresponds to the label the j'th text and k goes over P(j)

Algorithm 1: 
function TargetM: cap_m is the number of captions in the batch (number of text sentences). cls_m is the largest index of a label in the batch. In line 13, we simply assign, for each text caption, a label between cls_m + 1 and cap_m + cls_m + 1

Table 1 - in Cifar100 the vocab size is 105 (and not 100) since there are labels with more than one word (vocab size is number of words). In imagenet 1K its a bit more than 1K for the same reason. In imagenet 22-k ›it's less, I guess since there are repeating words. 


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

[13] ViT

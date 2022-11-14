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

As the title suggests, here I'll survey random interesting papers I came across. There's no real common theme connecting the papers, aside from me finding them really cool and having published this month 🤷‍♂️. I do try to give a relatively detailed overview so the reader can get the gist of work, but I definitely may skip things and possibly made errors, so feel free to comment, especially if you're by any chance one of the authors. 

CAN - A simple, efficient and scalable contrastive masked autoencoder for learning visual representations
======
[Arxiv] [code] subjects: semi-supervised learning, contrastive learning

Very cool paper that combines several ideas from self supervised learning (SSL), namely contrastive loss (most notably, SimCLR [1]) reconstruction of masked patches (most notably "Masked Autoencoders Are Scalable Vision Learners" [2]) and denoising autoencoder. Their pipeline works as follows: take an input image, generate 2 different views by applying augmentations, mask 50% of the patches, add Gaussian noise to the unmasked patches and feed the resulting noisy masked image to a ViT encoder. Now, we take the encoding of the unmasked patches, perform mean pooling, pass to a light MLP and perform contrastive learning, which gives us our contrastive loss. The encoded "patch image" is passed to a ViT decoder to perform reconstruction of the masked patches and denoising of the unmasked noisy patches, which gives us the reconstruction loss and the denoising loss. I'm skipping over a few technical details, such as feeding the noise level to the decoder and learning a "mask" indicator token.

The results demonstrate improved or on-par performance with recent SSL methods as measured on ImageNet 1K when finetuning or when using linear probing, both with pre-training on JFT-300 and pre-training on Imagenet. The result shows that it scales well to JFT-300, hence the "scalable" part of the title. The method is also faster than methods which use the full image views, as it only "uses" 50% of the tokens in both views of the image (as opposed to SimCLR for example) and does not use multiple views per image (such as DINO [3] or SwAV [4] which uses multi-crop to reduce the additional memory use), hence the "efficient" in the title. The paper is overall simple and elegant, does not use momentum encoder, hence the "simple" in the title. I should point out that it does not beat all other methods on all datasets, but the overall trade-off between results and simplicity is very good in my opinion and I also really like the combination of the different "trends" in SSL. 




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


References
======

[1] Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International conference on machine learning. PMLR, 2020.

[2] He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[3] Caron, Mathilde, et al. "Emerging properties in self-supervised vision transformers." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[4] Caron, Mathilde, et al. "Unsupervised learning of visual features by contrasting cluster assignments." Advances in Neural Information Processing Systems 33 (2020): 9912-9924.

[5] 


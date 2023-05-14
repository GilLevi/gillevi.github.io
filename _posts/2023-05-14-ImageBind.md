---
title: 'ImageBind: One Embedding Space To Bind Them All'
date: 2023-05-14
permalink: /posts/ImageBind
tags:
  - Deep Learning
  - Computer Vision
  - Research
  - Papers
  - Vision and Language
  - CLIP
  - NLP
  - Muti-modal 
---

The proposed idea is straightforward - mapping six different modalities to a joint embedding space. This allows data samples from different modalities, which share the same semantic meaning, to be mapped to similar vectors (i.e., vectors that are close in the cosine similarity metric) within the embedding space. Embedding modalities using explicitly aligned training data has been proposed before. For instance, the seminal works of CLIP[1] and ALIGN[2] map images and text to a joint embedding space. AudioCLIP[3] extends CLIP by adding an audio modality, and "Everything at Once"[4] embeds audio, images, and text into a joint mapping space to leverage the audio modality in video retrieval. However, embedding six different modalities - images, text, audio, depth, thermal, and Inertial Measurement Unit (IMU) data - particularly those without explicitly aligned training data or even datasets where they naturally coexist (e.g., you probably wouldn't find dataset of depth images associated with sounds) is a challenge that hasn't been tackled before. In my opinion, this opens the door to many new applications.

Blog https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/ 
Paper https://arxiv.org/abs/2305.05665 
Code https://github.com/facebookresearch/ImageBind 
Video https://dl.fbaipublicfiles.com/imagebind/imagebind_video.mp4 
Demo: https://imagebind.metademolab.com/demo 

Figure 1

In greater detail, the authors utilize the visual modality as the common link between the modalities by aligning each modality's embeddings to those of the images. For instance, IMU data is aligned with video using video data captured from egocentric cameras equipped with IMU. Here, the embeddings of the images and the second modality are learned using InfoNCE loss[5].

Consider a pair of modalities (I,M), where I represents images and M represents another modality. We learn two mapping functions, f and g, where f operates on images and g operates on the other modality. Given an image I_i and its corresponding data sample in the other modality M_i, we apply f to I_i and g to M_i to obtain the normalized embeddings qi = f(Ii) and ki = g(Mi). Both the encoders and the embeddings are learned by optimizing the InfoNCE loss[5]:

**InfoNCE equation**


There t is a scalar controlling the temperature and j denotes unrelated data samples in the batch, where every j =! i  is considered a negative pair. Optimizing this loss makes q_i and k_i close in the embedding space and optimizing on a large data set thus aligns the two modalities I and M. In practice, the symmetric loss L = L_I,M + L_M,I is used. 

This formulation assumes that for every modality M, we have a large-scale dataset with corresponding pairs (I_i, M_i). This is true for text, as large-scale text-image datasets have been collected (notably by the LAION Foundation[6,7,8]), but not for the other four modalities. However, such pairings naturally arise from other existing datasets. To this end, the authors used the (video, audio) pairs from the Audioset dataset [9], the (image, depth) pairs from the SUN RGB-D dataset [10], the (image, thermal) pairs from the LLVIP dataset [11], and the (video, IMU) pairs from the Ego4D dataset [12]. Note that ImageBind optimizes using only pairs of images and samples from other modalities, i.e., pairs of the form (I,M). The model is never explicitly trained to align other (M1, M2) pairs of modalities, such as learning to align depth maps with text. However, by optimizing the two "connections" (I, M1) and (I, M2), the alignment of M1 and M2 naturally emerges (similar to transitivity in math), allowing the model to perform zero-shot cross-modality retrieval between any pair of modalities.

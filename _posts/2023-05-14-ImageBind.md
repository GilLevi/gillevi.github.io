---
title: 'ImageBind: One Embedding Space To Bind Them All'
date: 2022-12-5
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

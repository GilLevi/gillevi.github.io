---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if site.author.googlescholar %}
  You can also find my articles on <u><a href="{{site.author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}


{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}


## [Temporal Tessellation: A Unified Approach for Video Analysis](https://talhassner.github.io/home/publication/2017_ICCV_2)
Dotan Kaufman, Gil Levi, Tal Hassner, and Lior Wolf. <i> Temporal Tessellation: A Unified Approach for Video Analysis.</i> International Conference on Computer Vision (ICCV), 2017.

## [The CUDA LATCH binary descriptor: because sometimes faster means better](https://talhassner.github.io/home/publication/2016_ECCV)
Christopher Parker, Matthew Daiter, Kareem Omar, Gil Levi and Tal Hassner. <i> The CUDA LATCH Binary Descriptor: Because Sometimes Faster Means Better. </i> Workshop on Local Features: State of the art, open problems and performance evaluation, at the European Conference on Computer Vision (ECCV), 2016.

## [LATCH: learned arrangements of three patch codes](https://talhassner.github.io/home/publication/2016_WACV_2)
Gil Levi and Tal Hassner.<i> LATCH: Learned Arrangements of Three Patch Codes. </i> IEEE Winter Conference on Applications of Computer Vision (WACV), 2016.

## [Emotion Recognition in the Wild via Convolutional Neural Networks and Mapped Binary Patterns](https://talhassner.github.io/home/publication/2015_ICMI)
Gil Levi and Tal Hassner. <i> Emotion Recognition in the Wild via Convolutional Neural Networks and Mapped Binary Patterns. </i> Proc. ACM International Conference on Multimodal Interaction (ICMI), Seattle, 2015.




---
menu:
- main
- footer
title: Resume
---

# Yiheng Li

- Deep learning application in medical imaging
- Multimodal data fusion
- Image registration and quality control
- Self-supervised learning
- Open source software development
- LLM, LoRA, Stable diffusion and AIGC in general

# EXPERIENCES

## Deep Learning Research Scientist at Subtle Medical Inc. 
> Menlo Park, CA 08/2021-07/2023
### Deep Learning based Image Co-Registration Quality Control
> 08/2021-02/2022
- Built a self-supervised classification model to QC the co-registration 
performance for pairs of 3D MRI images.
- Developed a data augmentation pipeline including affine and 
deformable movement to generate pseudo data pairs to tackle the data 
challenges of obtaining labelled well-registered MRI data pairs.
- To achieve multi-contrast prediction, compared two technical 
approaches: augmented contrasts vs. modality invariant representation.

*Meeting accepted:
[ISMRM 2022](https://submissions.mirasmart.com/ISMRM2022/Itinerary/ConferenceMatrix.aspx), ASNR 2022, RSNA 2022*

### Deep Learning-Based Multi Contrast MRI Registration Model with a Realistic Flow Field and Reduced Over-Smoothing Effect
> 02/2022-09/2022
- Developed a deep learning based co-registration model that can be 
applied to multi-contrast MRI images of multiple anatomies.
- Novel attempts, using “Jacobin loss” and “cycle consistent loss”, to deal 
with unrealistic flow field and over-smoothing effect of deep learning-
based registration methods, especially in the “VoxelMorph” and 
“SynthMorph” framework.
- Converted TensorFlow based “SynthMorph” code and trained PyTorch 
versions of “SynthMorph” and other variations of the model.
- Improved the SSIM and PSNR of the registered image on BraTS and 
Lumbar-Spine Open Dataset by ~40% and ~50% respectively.

Meeting accepted: [ISMRM 2023](https://submissions.mirasmart.com/ISMRM2023/Itinerary/ConferenceMatrix.aspx)

### A Self-Supervised Key Point Detection Framework For Multiple Applications
> 09/2022-07/2023
- Explore, adapted, developed and assessed multiple technical 
workflows for self-supervised key point detection in medical images.
- Developed a two-step rule-based pipeline for brain MRI auto-
formatting, using ANTs affine registration and SIFT key point matching.
- Guiding a intern to adopted and optimized the performance of the 
“KeyMorph”, an automatic key point generator with registration training 
mechanism, by customizing the loss with distance and applying 
additional mask.
- Start with the replication of a real-time self-supervised key point 
detector paper in PyTorch Lightning. Optimize over the original paper’s 
result by ~30% on a private self-curated test dataset with modified 
training strategy and grid loss search. A multi-purpose training 
framework is introduced by adding a tail to provide binary prediction of 
the key point existence. Transformer-based encoder replaced the CNN 
based encoder. Reinforced learning and iterative predictions are both 
tested for finer prediction.
### Creation and Maintenance of a PyTorch-Lightning and MONAI based Deep Learning Training, Logging and Inference Helper Package: `Lumos-ToolKit`
> 09/2022-07/2023
- Created and maintained a toolkit which includes pipelines for the key 
steps and pain point in deep learning model development for medical 
imaging: dataset curation and testing; image affine, deformable and 
other spatial transformations; logging and documentation of the model 
hyper-parameters, figures and performances; management of the 
training settings and configurations; image preprocessing pipeline; 
complicated loss settings and combinations of losses.
- The whole package dynamically integrates the following packages: 
“pydantic”, “argdantic”, “PyTorch-Lightning”, “MONAI”, “rich”, etc.

# Education

### Stanford University
> 09/2019-06/2021

*M.Sc. in Biomedical Informatics*

### Shanghai Jiao Tong University 
> 09/2015-06/2019

*B.Sc. in Resource and Environmental Science*  

### University of California, Berkeley
> 01/2018-05/2018 

*International Exchange Program*
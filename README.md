# Restoring-Extremely-Dark-Images-In-Real-Time

> The project is the official implementation of our *[CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lamba_Restoring_Extremely_Dark_Images_in_Real_Time_CVPR_2021_paper.pdf), "Restoring Extremely Dark Images in Real Time"*<br>  **&mdash; [Mohit Lamba](https://mohitlamba94.github.io/about-me/), Kaushik Mitra**

<p align="center">
  <a href="https://youtu.be/z22BuOb1igY">
  <img src="https://raw.githubusercontent.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time/main/imgs/youtube_cvpr.jpeg" alt="Click to watch Demo Video" height="380">
  </a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time/main/imgs/comparison.png" height="380">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time/main/imgs/table.png" height="320">
</p>

A practical low-light enhancement solution must be **computationally fast**, **memory-efficient**, and achieve a **visually appealing restoration**. Most of the existing methods target restoration quality and thus compromise on speed and memory requirements, raising concerns about their **real-world deployability**. We propose a new deep learning architecture for extreme low-light single image restoration, which despite its fast & lightweight inference, produces a restoration that is perceptually at par with state-of-the-art computationally intense models. To achieve this, we do most of the processing in the higher scale-spaces, skipping the intermediate-scales wherever possible. Also unique to our model is the potential to process all the scale-spaces concurrently, offering an additional 30% speedup without compromising the restoration quality. Pre-amplification of the dark raw-image is an important step in extreme low-light image enhancement. Most of the existing state of the art methods need GT exposure value to estimate the pre-amplification factor, which is not practically feasible. Thus, we propose an amplifier module that estimates the amplification factor using only the input raw image and can be used **“off-the-shelf”** with pre-trained models without any fine-tuning. We show that our model can restore an ultra-high-definition **4K resolution image** in just **1 sec. on a CPU** and at **32 fps on a GPU** and yet maintain a competitive restoration quality. We also show that our proposed model, without any fine-tuning, generalizes well to cameras not seen during training and to subsequent tasks such as object detection.

# How to Use the Code?
The code was tested on `Ubuntu 16.04 LTS` with `PyTorch 1.4`. Apart from commonly used libraries you need to install [rawpy](https://pypi.org/project/rawpy/) `pip install rawpy`.
## Quick Demo
We are in process of getting all the prior permissions and copyrights from agencies that supported this work and this should complete soon. Latest by July 2021 the code will be made public and paper put up on arxiv. 

## Cite Us

```
@inproceedings{RealTimeDarkImageRestorationCvpr2021,
  title={Restoring Extremely Dark Images in Real Time},
  author={Lamba, Mohit and Mitra, Kaushik},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3487--3497},
  year={2021}
}
```

## License
Copyright © Mohit Lamba, 2021. Patent Pending. All rights reserved. Please see the [license file](https://github.com/MohitLamba94/Restoring-Extremely-Dark-Images-In-Real-Time/blob/main/LICENSE) for terms.

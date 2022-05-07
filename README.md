# Recurrent-Affine-Transformation-for-Text-to-image-Synthesis

Official Pytorch implementation for our paper [Recurrent-Affine-Transformation-for-Text-to-image-Synthesis](https://arxiv.org/abs/2204.10482) 

![image](https://user-images.githubusercontent.com/10735956/166232980-d68a28e3-2f36-490d-acc5-dbbdad46c87e.png)
###Results
![图片](https://user-images.githubusercontent.com/10735956/167243219-3d9a39f6-b38f-4012-9988-ab9058249112.png)

---
### Requirements
- python 3.8
- Pytorch 1.11.0+cu113
- easydict
- nltk
- scikit-image
- A 2080 TI (set nf=32 in *.yaml) or a 3090 32GB (set nf=64 in *.yaml)

Note that nf=32  produces a IS around 5.0 on CUB. To reproduce the final results， please use a GPU more than 32GB.
### Installation

Clone this repo.
```
git clone https://github.com/senmaoy/Recurrent-Affine-Transformation-for-Text-to-image-Synthesis.git
cd Recurrent-Affine-Transformation-for-Text-to-image-Synthesis/code/
```

### Datasets Preparation
1. Download the preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`
4. Download [flower](https://drive.google.com/file/d/1cL0F5Q3AYLfwWY7OrUaV1YmTx4zJXgNG/view?usp=sharing) dataset and extract the images to `data/flower/`

Note that flower dataset is a bit different from cub and coco with a standalone dataset processing script.

### Pre-trained text encoder
1. Download the [pre-trained text encoder](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V) for CUB and save it to `../bird/`
2. Download the [pre-trained text encoder](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ) for coco and save it to `../bird/`
3. Download the [pre-trained text encoder](https://drive.google.com/file/d/1Gb5jRhSN9QGgmACNnZvwJMbDLDuVqffp/view?usp=sharing) for flower and save it to `../bird/`

---
### Training

**Train RAT-GAN models:**
  - For bird dataset: `python main.py --cfg cfg/bird.yml`
  - For coco dataset: `python main.py --cfg cfg/coco.yml`
  - For flower dataset: `python main.py --cfg cfg/flower.yml`

- `*.yml` files are example configuration files for training/evaluation our models.

### Evaluating

**Dwonload Pretrained Model**
- [RAT-GAN for bird](https://drive.google.com/file/d/1Np4odfdNkgRursGeKmwVix3zLhiZfZUa/view?usp=sharing). Download and save it to `models/bird/`
- [RAT-GAN for coco](https://drive.google.com/file/d/1wQOpopmaCFz9XjSnvjb5ZOVq2F-elFhy/view?usp=sharing). Download and save it to `models/coco/`
- [RAT-GAN for flower](https://drive.google.com/file/d/19THxubZDsa6_KfOTBpZ45S2aeVXCBj0T/view?usp=sharing). Download and save it to `models/flower/`

**Evaluate RAT-GAN models:**

- To evaluate our RAT-GAN on CUB, change B_VALIDATION to True in the bird.yml. and then run `python main.py --cfg cfg/bird.yml`
- To evaluate our RAT-GAN on coco, change B_VALIDATION to True in the coco.yml. and then run `python main.py --cfg cfg/coco.yml`
- We compute inception score for models trained on birds using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).
- We compute FID for CUB and coco using (https://github.com/senmaoy/Inception-Score-FID-on-CUB-and-OXford.git). 

---
### Citing RAT-GAN

If you find RAT-GAN useful in your research, please consider citing our paper:

```
@article{ye2022recurrent,
  title={Recurrent Affine Transformation for Text-to-image Synthesis},
  author={Ye, Senmao and Liu, Fei and Tan, Minkui},
  journal={arXiv preprint arXiv:2204.10482},
  year={2022}
}
```
The code is released for academic research use only. 

I'm currently focusing on GAN and GLOW models. If you are interseted, just contact me through senmaoy@gmail.com or Wechat: Unsupervised2020


**Reference**
- [DF-GAN:  DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis](https://arxiv.org/abs/2008.05865) [[code]](https://github.com/tobran/DF-GAN.git)
- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916) [[code]](https://github.com/hanzhanggit/StackGAN-v2)
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) [[code]](https://github.com/taoxugit/AttnGAN)
- [DM-GAN: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1904.01310) [[code]](https://github.com/MinfengZhu/DM-GAN)

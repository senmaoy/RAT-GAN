# Recurrent-Affine-Transformation-for-Text-to-image-Synthesis
a conditioanl GAN based text-to-image model
![image](https://user-images.githubusercontent.com/10735956/166214549-27fe0261-915d-45d4-83ab-d737f61ba672.png)
![image](https://user-images.githubusercontent.com/10735956/166214600-a09bcb2d-b786-44d0-8082-1a968d2765b4.png)
![image](https://user-images.githubusercontent.com/10735956/166214622-a5c5310d-273a-4f7f-8b2c-c55c9e7c0627.png)

Paper:https://arxiv.org/abs/2204.10482

Requirements
python 3.6+
Pytorch 1.0+
easydict
nltk
scikit-image
A titan xp (set nf=32 in *.yaml) or a 3090 32GB (set nf=64 in *.yaml)



Installation:
Just clone the code and run

Data:
refer to DF-GAN https://github.com/tobran/DF-GAN

Training


For bird dataset: python main.py --cfg cfg/bird.yml

For coco dataset: python main.py --cfg cfg/coco.yml

*.yml files are example configuration files for training/evaluation our models.

Pre-trained models
release soon in two or three days.

If you find RAT-GAN useful in your research, please consider citing our paper:


@article{ye2022recurrent,
  title={Recurrent Affine Transformation for Text-to-image Synthesis},
  author={Ye, Senmao and Liu, Fei and Tan, Minkui},
  journal={arXiv preprint arXiv:2204.10482},
  year={2022}
}

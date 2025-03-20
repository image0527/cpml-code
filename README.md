# CPML: Clustered Prototypical Meta-Learning Framework for Few-Shot Learning

This repository contains the code for: CPML: Clustered Prototypical Meta-Learning Framework for Few-Shot Learning<br>
<img src="https://github.com/image0527/cpml-code/blob/main/CPML-v2.png" />
Our code is based on [few-shot-meta-baseline](https://github.com/yinboc/few-shot-meta-baseline).

## Requirements

* python == 3.8.10
* cuda == 11.1
* torch == 1.9.0
* torchvision == 0.10.0
* numpy == 1.21.2
* Pillow == 8.3.2
* scikit-learn == 1.3.2	
* scipy == 1.10.1	
* tensorboard == 2.6.0	
* tensorboardX == 2.6.2.2	
* matplotlib == 3.4.3
## Running the code

### 1. Training Classifier-Baseline

```
python train_classifier.py --config configs/train_classifier_mini.yaml
```

(The pretrained Classifier-Baselines can be downloaded [here](https://www.dropbox.com/sh/ef2sm8d8qadhg3a/AAAIBotzaCKIdN1dJTvgDk-wa?dl=0))

### 2. Training Meta-Baseline

```
python train_meta.py --config configs/train_meta_mini.yaml
```

### 3. Test

To test the performance, modify `configs/test_few_shot.yaml` by setting `load_encoder` to the saving file of Classifier-Baseline, or setting `load` to the saving file of Meta-Baseline.

E.g., `load: ./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/max-va.pth`

Then run

```
python test_few_shot.py --shot 1
```

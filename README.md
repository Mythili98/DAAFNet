# EEG-EmotionRecognition
Dataset used in the study: download DEAP dataset from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/.

Use analysis-plots.ipynb to understand the underlying distribution of the fused connectivity matrix, which sheds light into how the model would perform on those features.

BC-Model1 - CNN Model without any domain adaptation


BC-AugModel - CNN Model with time series augmentation


eeg_da_skf - Domain adaptive model (Addition to BC-Model1) - Baseline


eeg_da_skf_Augmodel - Domain Adaptive Model(Addition to BC-AugModel1) - DAAFNet


SHAP_DAAFNET.ipynb - For model interpretability

## **Model Architecture**
![image](https://github.com/Mythili98/EEG-EmotionRecognition-DA-AFNet/assets/36411676/fab242c1-170a-405e-abad-bfcc1d943ec6)

### Architecture Details
from torchinfo import summary

summary(model.to(device=devices),[(1,5,32,32),(1,1,384,32)])

___________________________________________________________________________
```plaintext
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNNModel                                 [1, 2]                    --
├─Sequential: 1-1                        [1, 128, 15, 15]          --
│    └─Conv2d: 2-1                       [1, 32, 34, 34]           1,472
│    └─ResidualBlock: 2-2                [1, 32, 34, 34]           --
│    │    └─BatchNorm2d: 3-1             [1, 32, 34, 34]           64
│    │    └─ReLU: 3-2                    [1, 32, 34, 34]           --
│    │    └─Conv2d: 3-3                  [1, 32, 34, 34]           25,632
│    │    └─BatchNorm2d: 3-4             [1, 32, 34, 34]           64
│    │    └─ReLU: 3-5                    [1, 32, 34, 34]           --
│    │    └─Conv2d: 3-6                  [1, 32, 34, 34]           25,632
│    └─ResidualBlock: 2-3                [1, 64, 34, 34]           --
│    │    └─BatchNorm2d: 3-7             [1, 32, 34, 34]           64
│    │    └─ReLU: 3-8                    [1, 32, 34, 34]           --
│    │    └─Conv2d: 3-9                  [1, 64, 34, 34]           51,264
│    │    └─BatchNorm2d: 3-10            [1, 64, 34, 34]           128
│    │    └─ReLU: 3-11                   [1, 64, 34, 34]           --
│    │    └─Conv2d: 3-12                 [1, 64, 34, 34]           102,464
│    └─ResidualBlock: 2-4                [1, 128, 34, 34]          --
│    │    └─BatchNorm2d: 3-13            [1, 64, 34, 34]           128
│    │    └─ReLU: 3-14                   [1, 64, 34, 34]           --
│    │    └─Conv2d: 3-15                 [1, 128, 34, 34]          204,928
│    │    └─BatchNorm2d: 3-16            [1, 128, 34, 34]          256
│    │    └─ReLU: 3-17                   [1, 128, 34, 34]          --
│    │    └─Conv2d: 3-18                 [1, 128, 34, 34]          409,728
│    └─AdaptiveAvgPool2d: 2-5            [1, 128, 15, 15]          --
├─Sequential: 1-2                        [1, 192]                  --
│    └─EEGNet: 2-6                       [1, 192]                  --
│    │    └─Conv2d: 3-19                 [1, 16, 384, 1]           528
│    │    └─BatchNorm2d: 3-20            [1, 16, 384, 1]           32
│    │    └─ZeroPad2d: 3-21              [1, 1, 17, 417]           --
│    │    └─Conv2d: 3-22                 [1, 4, 16, 386]           260
│    │    └─BatchNorm2d: 3-23            [1, 4, 16, 386]           8
│    │    └─MaxPool2d: 3-24              [1, 4, 4, 97]             --
│    │    └─ZeroPad2d: 3-25              [1, 4, 11, 100]           --
│    │    └─Conv2d: 3-26                 [1, 4, 4, 97]             516
│    │    └─BatchNorm2d: 3-27            [1, 4, 4, 97]             8
│    │    └─MaxPool2d: 3-28              [1, 4, 2, 24]             --
├─Sequential: 1-3                        [1, 2]                    --
│    └─Linear: 2-7                       [1, 1024]                 29,688,832
│    └─BatchNorm1d: 2-8                  [1, 1024]                 2,048
│    └─ReLU: 2-9                         [1, 1024]                 --
│    └─Dropout: 2-10                     [1, 1024]                 --
│    └─Linear: 2-11                      [1, 512]                  524,800
│    └─BatchNorm1d: 2-12                 [1, 512]                  1,024
│    └─ReLU: 2-13                        [1, 512]                  --
│    └─Linear: 2-14                      [1, 2]                    1,026
│    └─Softmax: 2-15                     [1, 2]                    --
├─Sequential: 1-4                        [1, 2]                    --
│    └─Linear: 2-16                      [1, 1024]                 29,688,832
│    └─BatchNorm1d: 2-17                 [1, 1024]                 2,048
│    └─ReLU: 2-18                        [1, 1024]                 --
│    └─Linear: 2-19                      [1, 2]                    2,050
│    └─Softmax: 2-20                     [1, 2]                    --
==========================================================================================
Total params: 60,733,836
Trainable params: 60,733,836
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.01
==========================================================================================
Input size (MB): 0.07
Forward/backward pass size (MB): 8.25
Params size (MB): 242.94
Estimated Total Size (MB): 251.26
==========================================================================================
```

### References for EEGNet and Domain Adaptation
EEGNet : paper - https://arxiv.org/abs/1611.08024, pytorch implementation - https://github.com/aliasvishnu/EEGNet/tree/master

Domain Adaptation : paper - http://sites.skoltech.ru/compvision/projects/grl/ , pytorch implementation - https://github.com/fungtion/DANN


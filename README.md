## Bilateral Recursive Network for Single Image Deraining
### Introduction
We in this paper propose a bilateral recurrent network (**BRN**) to simultaneously exploit the rain streak layer and the clean background image. 
Generally, we employ dual residual networks (**ResNet**) that are recursively unfolded to sequentially extract the rain streak layer (**Fr**) and predict the clean background image (**Fx**). 
In particular, we further propose bilateral LSTMs (**BLSTM**), which not only can respectively propagate deep features of the rain streak layer and the background image acorss stages, but also bring reciprocal communications between Fr and Fx. 
The experimental results demonstrate that our BRN notably outperforms state-of-the-art deep deraining networks on synthetic datasets quantitatively and qualitatively. On real rainy images, our BRN also performs more favorably in generating visually plausible background images. 


## Prerequisites
- Python 3.6, PyTorch >= 0.4.0
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)
- MATLAB for computing [evaluation metrics](statistics/)


## Datasets

BRN and its variants are evaluated on three datasets*: 
Rain100H [1], Rain100L [1] and Rain12 [2]. 
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), 
and place the unzipped folders into `./datasets/test/`.

To train the models, please download training datasets: 
RainTrainH [1] and RainTrainL [1] from [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg)
or [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g), 
and place the unzipped folders into `./datasets/train/`. 

*_We note that:_

_(i) The datasets in the website of [1] seem to be modified. 
    But the models and results in recent papers are all based on the previous version, 
    and thus we upload the original training and testing datasets 
    to [BaiduYun](https://pan.baidu.com/s/1J0q6Mrno9aMCsaWZUtmbkg) 
    and [OneDrive](https://1drv.ms/f/s!AqLfQqtZ6GwGgep-hgjLxkov2SSZ3g)._ 

_(ii) For RainTrainH, we strictly exclude 546 rainy images that have the same background contents with testing images.
    All our models are trained on remaining 1,254 training samples._
        

## Getting Started

### 1) Testing

We have placed our pre-trained models into `./logs/`. 

Run shell scripts to test the models:
```bash
bash test_Rain100H.sh   # test models on Rain100H
bash test_Rain100L.sh   # test models on Rain100L
bash test_Rain12.sh     # test models on Rain12
bash test_Rain1400.sh   # test models on Rain1400 
bash test_Ablation.sh   # test models in Ablation Study
bash test_real.sh       # test PReNet on real rainy images
```
All the results in the paper are also available at [BaiduYun](https://pan.baidu.com/s/1_La88cg4npzYpEv8Y6d5EQ).
You can place the downloaded results into `./results/`, and directly compute all the [evaluation metrics](statistics/) in this paper.  

### 2) Evaluation metrics

We also provide the MATLAB scripts to compute the average PSNR and SSIM values reported in the paper.
 

```Matlab
 cd ./statistic
 run statistic_Rain100H.m
 run statistic_Rain100L.m
 run statistic_Rain12.m
 run statistic_Rain1400.m
 run statistic_Ablation.m  # compute the metrics in Ablation Study
```
###
Average PSNR/SSIM values on four datasets:

Dataset    | BRN       |BRN-XR     |BRN-RX     |CRN  
-----------|-----------|-----------|-----------|-----------
Rain100H   |29.58/0.902|29.50/0.901|29.16/0.898|29.10/0.897
Rain100L   |37.82/0.981|37.65/0.980|37.40/0.979|37.52/0.980
Rain12     |36.70/0.959|36.63/0.959|36.54/0.959|36.58/0.959


### 3) Training

Run shell scripts to train the models:
```bash
bash train_PReNet.sh      
bash train_PRN.sh   
bash train_PReNet_r.sh    
bash train_PRN_r.sh  
```
You can use `tensorboard --logdir ./logs/your_model_path` to check the training procedures. 

### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 18            | Training batch size
recurrent_iter         | 6             | Number of recursive stages
epochs                 | 100           | Number of training epochs
milestone              | [30,50,80]    | When to decay learning rate
lr                     | 1e-3          | Initial learning rate
save_freq              | 1             | save intermediate model
use_GPU                | True          | use GPU or not
gpu_id                 | 0             | GPU id
data_path              | N/A           | path to training images
save_path              | N/A           | path to save models and status           

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0                | GPU id
recurrent_iter         | 6                | Number of recursive stages
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results

## References
[1] Yang W, Tan RT, Feng J, Liu J, Guo Z, Yan S. Deep joint rain detection and removal from a single image. In IEEE CVPR 2017.

[2] Li Y, Tan RT, Guo X, Lu J, Brown MS. Rain streak removal using layer priors. In IEEE CVPR 2016.

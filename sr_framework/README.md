# Requirements
* Python
    * numpy         1.19.1
    * python        3.7.8
* Pytorch
    * cudnn         7.6.5
    * pytorch       1.4.0
    * cudatoolkit   10.0.130
    * torchsummaryX 1.3.0
* Tensorflow
    * tensorflow    1.15.0
    * tensorboardX  2.1
* Image processing libraries
    * Pillow        7.2.0
    * imageio       2.9.0
    * opencv-python 4.3.0.36
* Data processing libraries
    * pyyaml        5.3.1
    * pandas        1.1.0
* Log and visualize libraries
    * tqdm          4.48.2
    * matplotlib    3.1.1

I also provide [requirements.yaml](https://github.com/NJU-Jet/SR\_Framework/blob/master/sr\_framework/requirements.yaml) for you to copy my conda environment. If you have anaconda, you can use the following codes:
* First make sure your nvidia driver version is larger than 410.48
```bash
nvidia-smi
```
* Create conda environment from yaml
```bash
conda env create --f requirements.yaml
source activate SR
```
If you want to figure out which file includes the libraries(take tensorboardX as an example), you can use:
```bash
grep -rn --color=auto 'tensorboardX'
```

# Train on DIV2K
* Download DIV2K dataset from [EDVR](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md#REDS), unpack the tar file to any place you want.
* Change ```dataroot_HR```and```dataroot_LR``` arguments in ```options/train/{model}.yaml```to the place where DIV2K images are located.(change {model} according to your need)
* Run(change {model} according to your need, --use_chop is for saving memory in validation stage):
```bash
python train.py --opt options/train/{model}.yaml --name {model}_bs16ps64lr2e-4_x2 --scale 2 --lr 2e-4 --bs 16 --ps 64 --gpu_ids 0 --use_chop
```

# Train on your own dataset
* Change ```dataroot_HR```and```dataroot_LR``` arguments in ```options/train/{model}.yaml```to the place where your images are located.(change {model} according to your need)
* Change ```mode``` in ```options/train/{model}.yaml to ```TrainLRHR```
* Run(change {model} according to your need, --use_chop is for saving memory in validation stage):
```bash
python train.py --opt options/train/{model}.yaml --name {model}_bs16ps64lr2e-4_x2 --scale 2 --lr 2e-4 --bs 16 --ps 64 --gpu_ids 0 --use_chop
```

# Test on Benchmark(Set5, Set14, B100, Urban100, Mango109)
* Download benchmark dataset from [EDVR](https://github.com/xinntao/EDVR/blob/master/docs/DatasetPreparation.md#REDS), unpack the tar file to any place you want.
* Change ```dataroot_HR```and```dataroot_LR``` arguments in ```options/test/base.yaml```to the place where benchmark images are located.
* Run():
```bash
python test.py --opt options/test/base.yaml --dataset_name {dataset_name} --scale {scale} --which_model {model} --pretrained {pretrained_path}
```
For example:
```bash
python test.py --opt options/test/base.yaml --dataset_name Set5 --scale 2 --which_model EDSR --pretrained pretrained/EDSR.pth
```

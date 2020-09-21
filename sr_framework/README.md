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

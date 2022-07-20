# UDC MIPI-Challenge

An implementation code for UDC MIPI-Challenge

## Introduction

The under-display camera (UDC) brings a bezel-less and notch-free viewing experience for smartphones. Meanwhile, photos captured by UDC suffer from severe degradation caused by the semi-transparent organic light-emitting diode (OLED) display mounted before the UDC. Zhou et al. pioneer the attempt of the UDC image restoration and propose a Monitor Camera Imaging System (MCIS) to capture paired data. However, their work only consider the images with low dynamic range shown on the monitor. To alleviate this problem, Feng et al. reformulate the image formation model and synthesis the UDC image by considering the diffraction flare of the saturated region in the high-dynamic-range images. In this challenge, we focus on developing an effective restoration algorithm for the synthetic UDC images. In order to reduce the storage consumption and computation complexity caused by self-attention mechanisms in transformer, we build our method based on a nonlinear activation free network (NAFNet).
## Requisites

* Pytorch 1.9.0
* Python 3
* Linux

## Test

### Prepare Test Data

Download the challenge test data and unzip the test set, and then copy them to `dataset`.

### Download Pre-trained Model

Download and unzip our [pre-trained model](https://drive.google.com/file/d/1mTrTcvp0DTrG_xsvQK-8U16UTBxQ_DKK/view?usp=sharing), and then copy them to `checkpoints`.

### Run

You can run 
```python
python test.py -opt ./UDC_restor/Options/UDC_test_synthetic_data_NAFNet.yml
```
The test results can be found in `results/UDC_test_synthetic_data_NAFNet/visualization/npy_new/`


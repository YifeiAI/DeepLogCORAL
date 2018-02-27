# DeepLogCORAL
Caffe implementation of DeepLogCORAL
Please see our paper for more detail: https://arxiv.org/pdf/1707.09842.pdf

Caffe version: 1.0.0-rc3 (haven't tested on caffe 2)

## Setup: 

add files in layers into your caffe/src/caffe/layers/

add files in include into your caffe/include/caffe/layers

recompile caffe

## Network structure:

![alt text](https://github.com/YifeiAI/DeepLogCORAL/blob/master/image/structure.png)

## Demo training curve (source: Amazon  target: DSLR):

![alt text](https://github.com/YifeiAI/DeepLogCORAL/blob/master/image/AW_two.png)

#!/usr/bin/env sh

#MXNET_ENABLE_GPU_P2P=0 MXNET_CUDNN_AUTOTUNE_DEFAULT=0 nohup python -u train.py  > 0508_davis_youtube_512.log 2>&1 &
nohup python -u train.py  > 0524.log 2>&1 &

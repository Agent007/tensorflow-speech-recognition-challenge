#!/bin/bash -e

export CUDA_VISIBLE_DEVICES=$1
source activate root
python predict.py

#!/bin/bash -e

source activate root
export CUDA_VISIBLE_DEVICES=$1
python predict.py
